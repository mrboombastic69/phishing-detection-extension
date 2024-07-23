import torch
import torch.nn as nn
import joblib
import numpy as np
import re
import whois
from datetime import datetime
import requests
import dns.resolver
import random
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urlsplit
from sklearn.preprocessing import MinMaxScaler

# Define the model class
class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.layer_1 = nn.Linear(23, 300)
        self.layer_2 = nn.Linear(300, 100)
        self.layer_out = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(300)
        self.batchnorm2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        x = self.sigmoid(x)
        return x
