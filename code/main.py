import json
import os
import random
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm.notebook import tqdm, trange
from transformers import AutoModel, AutoTokenizer, get_scheduler
import torch.nn as nn
from torchvision.models.resnet import resnet50
import torchvision.transforms as transforms


HOME_FOLDER = os.getcwd()
DATA_FOLDER = os.path.join(HOME_FOLDER, 'data')
IMG_FOLDER = HOME_FOLDER
SENTENCE_FOLDER = DATA_FOLDER
RESULTS_FOLDER = os.path.join(HOME_FOLDER, 'results')
TRAIN_DATA_LABEL = './train.txt'
os.makedirs(RESULTS_FOLDER, exist_ok=True)

df_train = pd.read_csv(os.path.join(HOME_FOLDER, 'train.csv'))
df_train, df_dev = train_test_split(df_train, test_size=0.2, random_state=42)

num_train_epochs = 50
batch_size = 4
learning_rate = 1.0e-5
weight_decay = 0.01
warmup_steps = 0
max_seq_length = df_train['text'].apply(len).max()

print(f"Hyperparameters \nepoch: {num_train_epochs}\nbatch_size: {batch_size}\nlr: {learning_rate}\nweight_decay: {weight_decay}\nwarmup_steps: {warmup_steps}\nmax_seq_length: {max_seq_length}")

label_to_id = {lab: i for i, lab in enumerate(df_train['tag'].sort_values().unique())}
id_to_label = {v: k for k, v in label_to_id.items()}
num_label = len(label_to_id)

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

class TextDataset(Dataset):
    def __init__(self, df, label_to_id, text_field="text", label_field="tag"):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.text_field = text_field
        self.label_field = label_field

    def __getitem__(self, index):
        text = str(self.df.at[index, self.text_field])
        label = self.label_to_id.get(self.df.at[index, self.label_field], -1)
        return text, label

    def __len__(self):
        return self.df.shape[0]

class BertModel(nn.Module):
    def __init__(self, num_labels, text_pretrained='./bert-base-uncased'):
        super().__init__()
        self.num_labels = num_labels
        self.text_encoder = AutoModel.from_pretrained(text_pretrained)
        self.classifier = nn.Linear(self.text_encoder.config.hidden_size, num_labels)

    def forward(self, text):
        output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        logits = self.classifier(output.last_hidden_state[:, 0, :])  # CLS embedding
        return logits

class ResNetDataset(Dataset):
    def __init__(self, df, label_to_id, train=False, text_field="text", label_field="tag", image_path_field="img_path"):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.train = train
        self.text_field = text_field
        self.label_field = label_field
        self.image_path_field = image_path_field

        # ResNet-50 settings
        self.img_size = 224
        self.mean, self.std = (
            0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


        self.train_transform_func = transforms.Compose(
                [transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])

        self.eval_transform_func = transforms.Compose(
                [transforms.Resize(256),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                    ])


    def __getitem__(self, index):
        text = str(self.df.at[index, self.text_field])
        label = self.label_to_id.get(self.df.at[index, self.label_field], -1)
        # self.label_to_id[self.df.at[index, self.label_field]]
        img_path = self.df.at[index, self.image_path_field]


        image = Image.open(IMG_FOLDER + '/' + img_path)
        if self.train:
          img = self.train_transform_func(image)
        else:
          img = self.eval_transform_func(image)

        return text, label, img

    def __len__(self):
        return self.df.shape[0]

class FeatureModel(nn.Module):
    def __init__(self, output_layer):
        super().__init__()
        self.output_layer = output_layer
        pretrained_resnet = resnet50(pretrained=True)
        self.children_list = []
        for n,c in pretrained_resnet.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break

        self.net = nn.Sequential(*self.children_list)


    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return x

class ResNetModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.visual_encoder = FeatureModel(output_layer='avgpool')
        self.image_hidden_size = 2048

        self.classifier = nn.Linear(self.image_hidden_size, num_labels)

    def forward(self, text, image):
        img_feature = self.visual_encoder(image)
        features = torch.cat((img_feature), 1)

        logits = self.classifier(features)

        return logits
class MML(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size1 + input_size2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
class MMLmodel(nn.Module):
    def __init__(self, num_labels, text_pretrained='./bert-base-uncased', hidden_size=512):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_pretrained)
        self.visual_encoder = FeatureModel(output_layer='avgpool')
        self.image_hidden_size = 2048
        self.mlp_text = nn.Linear(self.text_encoder.config.hidden_size, num_labels)
        self.mlp_image = nn.Linear(self.image_hidden_size, num_labels)
        self.mlp_both = MML(self.text_encoder.config.hidden_size, self.image_hidden_size, hidden_size, num_labels)

    def forward(self, text, image, mode='both'):
        text_output = self.text_encoder(**text)
        text_feature = text_output.last_hidden_state[:, 0, :]
        img_feature = self.visual_encoder(image)

        if mode == 'text':
            logits = self.mlp_text(text_feature)
        elif mode == 'image':
            logits = self.mlp_image(img_feature)
        else:  # 'both'
            logits = self.mlp_both(text_feature, img_feature)

        return logits

bert_tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased')
tokenizer_path = bert_tokenizer.save_pretrained('/proj5')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model = BertModel(num_labels=num_label)
bert_model = bert_model.to(device)
bert_resnet_model = MMLmodel(num_labels=num_label, text_pretrained='./bert-base-uncased')
bert_resnet_model = bert_resnet_model.to(device)

def run_mml(epochs, mode='both'):
  num_train_epochs = epochs
  seed_val = 42
  set_seed(seed_val)

  set_seed(seed_val)

  train_dataset = ResNetDataset(df=df_train, label_to_id=label_to_id, train=True, text_field='text', label_field='tag', image_path_field='img_path')
  train_sampler = RandomSampler(train_dataset)
  train_dataloader = DataLoader(dataset=train_dataset,
                      batch_size=batch_size,
                      sampler=train_sampler)


  t_total = len(train_dataloader) * num_train_epochs


  optimizer = AdamW(bert_resnet_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

  criterion = nn.CrossEntropyLoss()

  bert_resnet_model.train()

  start = perf_counter()
  for epoch_num in trange(num_train_epochs, desc='Epochs'):
      epoch_total_loss = 0

      for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):
          b_text, b_labels, b_imgs = batch
          b_inputs = bert_tokenizer(
              list(b_text), truncation=True, max_length=max_seq_length,
              return_tensors="pt", padding=True
          )

          b_labels = b_labels.to(device)
          b_imgs = b_imgs.to(device)
          b_inputs = b_inputs.to(device)

          bert_resnet_model.zero_grad()
          # b_logits = resnet_model(text=b_inputs, image=b_imgs)
          b_logits = bert_resnet_model(text=b_inputs, image=b_imgs, mode='both')

          loss = criterion(b_logits, b_labels)

          epoch_total_loss += loss.item()

          loss.backward()


          optimizer.step()
          scheduler.step()

      avg_loss = epoch_total_loss/len(train_dataloader)


      print('epoch =', epoch_num)
      print('    epoch_loss =', epoch_total_loss, end="\t")
      print('    avg_epoch_loss =', avg_loss, end="\t")
      print('    learning rate =', optimizer.param_groups[0]["lr"], end="\t")
  end = perf_counter()
  resnet_training_time = end- start
  print(resnet_training_time)

def val_mml(mode='both', file_download=False):
  resnet_validation_results = []

  val_dataset = ResNetDataset(df=df_dev, label_to_id=label_to_id, train=False, text_field='text', label_field='tag',
                              image_path_field='img_path')
  val_sampler = SequentialSampler(val_dataset)
  val_dataloader = DataLoader(dataset=val_dataset,
                              batch_size=batch_size,
                              sampler=val_sampler)

  for batch in tqdm(val_dataloader):
      bert_resnet_model.eval()

      b_text, b_labels, b_imgs = batch

      b_inputs = bert_tokenizer(list(b_text), truncation=True, max_length=max_seq_length, return_tensors="pt",
                                padding=True)

      b_labels = b_labels.to(device)
      b_imgs = b_imgs.to(device)
      b_inputs = b_inputs.to(device)

      with torch.no_grad():
          b_logits = bert_resnet_model(text=b_inputs, image=b_imgs, mode=mode)
          b_logits = b_logits.detach().cpu()

      resnet_validation_results += torch.argmax(b_logits, dim=-1).tolist()

  resnet_validation_labels = [id_to_label[p] for p in resnet_validation_results]
  resnet_class_report_val = classification_report(df_dev['tag'], resnet_validation_labels, output_dict=True)
  print(resnet_class_report_val['accuracy'])
  if file_download is True:
      with open(RESULTS_FOLDER + 'resnet_class_report.json', 'w') as f:
          json.dump(resnet_class_report_val, f)


# run_mml(epochs=1, mode='text')
# val_mml('text')

# run_mml(epochs=1, mode='img')
# val_mml('img')

run_mml(epochs=5, mode='both')
val_mml('both')


def predict_and_save_results(model, tokenizer, test_dataloader, id_to_label, device, result_file):
    model.eval()
    prediction_results = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            b_text, b_labels, b_imgs = batch

            b_labels = b_labels.to(device)
            b_imgs = b_imgs.to(device)

            b_inputs = tokenizer(list(b_text), truncation=True, max_length=max_seq_length, return_tensors="pt", padding=True)
            b_inputs = b_inputs.to(device)

            b_logits = model(text=b_inputs, image=b_imgs)
            b_logits = b_logits.detach().cpu()

            prediction_results += torch.argmax(b_logits, dim=-1).tolist()

    resnet_prediction_labels = [id_to_label[p] for p in prediction_results]

    df_test['tag'] = resnet_prediction_labels

    df_test.to_csv(result_file, index=False)

df_test = pd.read_csv(HOME_FOLDER + '/test.csv')

test_dataset = ResNetDataset(df=df_test, label_to_id=label_to_id, train=False, text_field='text', label_field='tag', image_path_field='img_path')
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=test_sampler)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_resnet_model.to(device)

id_to_label = {v: k for k, v in label_to_id.items()}

result_file = HOME_FOLDER + '/' + 'result.csv'
predict_and_save_results(bert_resnet_model, bert_tokenizer, test_dataloader, id_to_label, device, result_file)
