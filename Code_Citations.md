# Code Citations

## License: Apache

<https://github.com/pankaiqianghub/ODE/blob/1d43e317ddc71bccb0d0e49ed5de607f5ea313b8/ODE-main/MF/VAE.py>
<https://github.com/pankaiqianghub/ODET/blob/main/ODET-main/LICENSE>

```python
, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)
```

<https://github.com/pankaiqianghub/ODE/blob/1d43e317ddc71bccb0d0e49ed5de607f5ea313b8/ODE-main/MF/VAE.py>
<https://github.com/pankaiqianghub/ODET/blob/1d43e317ddc71bccb0d0e49ed5de607f5ea313b8/ODE-main/LICENSE>

```python
, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)
```

## License: Open-Source Data

<https://github.com/ethanswang/cancer-biomarker-project/blob/fb20566a8d6b53e6dc7f33f18786c4573721ec73/SCVAE.py>
built upon <https://ngdc.cncb.ac.cn/omix/release/OMIX001073> which uses Open-Source data accessibility.

```python
, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)
```

<https://github.com/ethanswang/cancer-biomarker-project/blob/fb20566a8d6b53e6dc7f33f18786c4573721ec73/SCVAE.py>
built upon <https://ngdc.cncb.ac.cn/omix/release/OMIX001073> which uses Open-Source data accessibility.

```python
, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)
```
