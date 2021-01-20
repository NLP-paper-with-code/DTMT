from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab

import torchtext
import torch

from collections import Counter

class ToTensor(object):
  def __call__(self, sample, source_vocab, target_vocab):    
    source, target = sample
    source_tensor = torch.tensor([source_vocab[word] for word in source.split()], dtype=torch.long)
    target_tensor = torch.tensor([target_vocab[word] for word in target.split()], dtype=torch.long)

    return source_tensor, target_tensor

class TranslationDataset(Dataset):
  def __init__(self, path, source_language, target_language, transform=None):
    """
    path: path of data
    source_language: source language, i.e., fr, en
    target_language: target language, i.e., fr, en
    """
    super().__init__()
    
    self.data = []
    self._source_vocab = None
    self._target_vocab = None
    self.transform = transform

    with open(f'{path}.{source_language}', 'r', encoding='utf-8') as source_language_file, \
      open(f'{path}.{target_language}', 'r', encoding='utf-8') as target_language_file:

      raw_source = source_language_file.readlines()
      raw_target = target_language_file.readlines()

      source_counter, target_counter = Counter(), Counter()

      for (source, target) in zip(raw_source, raw_target):
        source = '<sos>' + source.strip() + '<eos>'
        target = '<sos>' + target.strip() + '<eos>'
        
        self.data.append((source, target))
        
        source_counter.update([word for word in source.split()])
        target_counter.update([word for word in target.split()])

      self._source_vocab = Vocab(source_counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
      self._target_vocab = Vocab(target_counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    if self.transform:
      return self.transform(self.data[index], self.source_vocab, self.target_vocab)

    return self.data[index]
  
  @property
  def source_vocab(self):
    return self._source_vocab

  @property
  def target_vocab(self):
    return self._target_vocab

class PadBatch(object):
  def __init__(self, padding_value):
    self.padding_value = padding_value

  def __call__(self, batch):
    source_batch, target_batch = [], []

    for (source, target) in batch:
      source_batch.append(source)
      target_batch.append(target)

    source_batch = pad_sequence(source_batch, padding_value=self.padding_value)
    target_batch = pad_sequence(target_batch, padding_value=self.padding_value)

    return source_batch, target_batch

if __name__ == '__main__':
  base_path = f'wmt14_en_fr'
  test_path = f'{base_path}/test'

  test_dataset = TranslationDataset(test_path, source_language='fr', target_language='en', transform=ToTensor())

  test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=PadBatch(padding_value=test_dataset.source_vocab['<pad>']))

  for test in test_dataloader:
    print(test)