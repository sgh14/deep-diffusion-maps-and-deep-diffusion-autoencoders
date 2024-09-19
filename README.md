# deep-diffusion-maps-and-deep-diffusion-autoencoders
Implementation of Diffusion Maps using neural networks and combination of Diffusion Maps with Deep Autoencoders.

## Example

```python
from sklearn.datasets import make_swiss_roll
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from DiffusionLoss import DiffusionLoss
from DeepDiffusionAE import DeepDiffusionAE

X, color = make_swiss_roll(1000)
n_features = X.shape[1]
n_components = 2
units = 128
```

### Deep Diffusion Maps

```python
encoder = Sequential([
    Input(shape=(n_features,)),
    Dense(units, activation='relu'),
    Dense(units//2, activation='relu'),
    Dense(units//4, activation='relu'),
    BatchNormalization(),
    Dense(n_components, use_bias=False, activation='linear')
], name='encoder')

encoder.compile(optimizer='adam', loss=DiffusionLoss(X, sigma=1.7, t=100))
indices = np.array(list(range(X.shape[0])))
history_enc = encoder.fit(
    x=X,
    y=indices,
    epochs=500,
    batch_size=512,
    shuffle=True,
    validation_split=0.2
)

X_red = encoder(X)
```

### Deep Diffusion Autoencoder

```python
encoder = Sequential([
    Input(shape=(n_features,)),
    Dense(units, activation='relu'),
    Dense(units//2, activation='relu'),
    Dense(units//4, activation='relu'),
    BatchNormalization(),
    Dense(n_components, use_bias=False, activation='linear')
], name='encoder')

decoder = Sequential([
    Input(shape=(n_components,)),
    Dense(units//4, activation='linear'),
    Dense(units//2, activation='relu'),
    Dense(units, activation='relu'),
    Dense(n_features, activation='linear')
], name='decoder')

diffusionae = DeepDiffusionAE(encoder, decoder)
diffusionae.compile(X, sigma=1.7, t=100, diffusion_weight=50)
history = diffusionae.fit(X, epochs=1000, batch_size=512, shuffle=True, validation_split=0.2)
X_red = diffusionae.encode(X)
X_rec = diffusionae.decode(X_red)
```

## Experiments
