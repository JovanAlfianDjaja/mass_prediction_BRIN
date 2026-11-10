from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Lambda, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2,l1

# Path lengkap ke file
file_path = 'C:/Users/jovan/Downloads/NN/mass8.txt'

# Membaca file
data = np.loadtxt(file_path)

neutron            = data[:,0]
proton             = data[:,1]
nomor_massa        = data[:,2]
NEO                = data[:,3]
ZEO                = data[:,4]
N_shell            = data[:,5]
Z_shell            = data[:,6]
vN                 = data[:,7]
vZ                 = data[:,8]
PF                 = data[:,9]

input0             = data[:,[0,1,2]] # Sebagai awal, berisi jumlah neutron, jumlah proton, dan nomor massa

mass_excess        = data[:,10] #dalam satuan keV

neutron            = np.array(neutron)
proton             = np.array(proton)
nomor_massa        = np.array(nomor_massa)
NEO                = np.array(NEO)
ZEO                = np.array(ZEO)
N_shell            = np.array(N_shell)
Z_shell            = np.array(Z_shell)
input0             = np.array(input0)
mass_excess        = np.array(mass_excess)
vZ                 = np.array(vZ)
vN                 = np.array(vN)
PF                 = np.array(PF)

nomor_massa23      = nomor_massa**(2/3)
asimetri           = (neutron-proton)/(nomor_massa+1e-20)
coulomb            = proton*(proton-1)/(nomor_massa**(1/3)+1e-20)

# Normalisasi Input
# neutron            = (neutron - np.min(neutron)) / (np.max(neutron) - np.min(neutron))
# proton             = (proton - np.min(proton)) / (np.max(proton) - np.min(proton))
# nomor_massa        = (nomor_massa - np.min(nomor_massa)) / (np.max(nomor_massa) - np.min(nomor_massa))
# nomor_massa23      = (nomor_massa23 - np.min(nomor_massa23)) / (np.max(nomor_massa23) - np.min(nomor_massa23))
# asimetri           = (asimetri - np.min(asimetri)) / (np.max(asimetri) - np.min(asimetri))
# coulomb            = (coulomb - np.min(coulomb)) / (np.max(coulomb) - np.min(coulomb))
# vN                 = (vN - np.min(vN)) / (np.max(vN) - np.min(vN))
# vZ                 = (vZ - np.min(vZ)) / (np.max(vZ) - np.min(vZ))
# PF                 = (PF - np.min(PF)) / (np.max(PF) - np.min(PF))

input = np.c_[proton,neutron]

X=input
y=mass_excess
scaler=RobustScaler() #scaler=joblib.load()
X=scaler.fit_transform(X)
y=y/1000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=30)#random_state=42, 27 (good)
# recommended random state = 30
print("Training set size:", X_train.shape, y_train.shape)
print("Test set size:", X_test.shape, y_test.shape)

joblib.dump(scaler, 'scaler2input.pkl')

y_train_normal  = y_train
y_test_normal   = y_test

reduce_lr = ReduceLROnPlateau(monitor='loss',  # Monitor validation loss val_loss
                              factor=0.75,         # Reduce learning rate by half
                              patience=20, #10         # Wait for 5 epochs before reducing LR
                              verbose=1,          # Print a message when reducing LR
                              mode='min',         # Reduce if the loss doesn't improve
                              min_lr=1e-30)        # Lower bound for learning rate

lr_schedule = reduce_lr

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(75, input_dim=n_inputs, kernel_initializer='lecun_normal',  activation='selu'))
    model.add(Dense(75, kernel_initializer='lecun_normal', activation='selu'))
    model.add(Dense(75, kernel_initializer='lecun_normal', activation='selu'))
    model.add(Dense(75, kernel_initializer='lecun_normal', activation='selu'))
    model.add(Dense(75, kernel_initializer='lecun_normal', activation='selu'))
    model.add(Dense(75, kernel_initializer='lecun_normal', activation='selu'))
    model.add(Dense(75, kernel_initializer='lecun_normal', activation='selu'))
    model.add(Dense(75, kernel_initializer='lecun_normal', activation='selu'))

    model.add(Dense(n_outputs, kernel_initializer='lecun_normal'))#'lecun_normal'))
#    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    opt = keras.optimizers.Nadam(learning_rate=0.001*14) #0.001*1.5
    model.compile(loss='mse', optimizer=opt)
    return model

# Early stopping untuk mencegah overfitting
early_stopping = EarlyStopping(min_delta=1e-5, patience=50, restore_best_weights=True)

n_inputs, n_outputs = X_train.shape[1], 1
model = get_model(n_inputs, n_outputs)

model.summary()

history = model.fit(
#    X_train, y_train,
    X_train, y_train_normal,
    validation_split=0.25,     # 0.25
    epochs=1000,
    batch_size=64,
    callbacks=[lr_schedule,early_stopping],
    verbose=1
)

#Plotting
plt.plot(history.history['loss'], color='orange')
#plt.plot(history.history['val_loss'])
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
#plt.legend(['Train', 'Val'], loc='upper right')
plt.legend(['Train'], loc='upper right')
plt.show()

#plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.legend(['Val'], loc='upper right')
plt.show()

#Mendapat nilai prediksi
mass_excess_pred = np.array([])
for i in range(X_test.shape[0]):
 input_X = X_test[i]
 #row = [input_X]
 new_row = input_X.reshape(1, -1)
 fit = model.predict(new_row)
 mass_excess_pred = np.append(mass_excess_pred, fit)

x0 = np.linspace(-10,100,1000)
y0 = x0
#plt.plot(y_test_MeV,mass_excess_MeV,'.')
plt.plot(y_test_normal,mass_excess_pred,'.')
plt.plot(x0,y0,linestyle='dashed')
plt.title('Neural Network, R^2=%.2f' % r2_score(y_test_normal, mass_excess_pred))
plt.ylabel('Predicted Mass Excess')
plt.xlabel('Data Mass Excess')
plt.show()

r2 = r2_score(y_test_normal,mass_excess_pred)
print("R2:", r2)

# Mean Squared Error (MSE)
#mse = np.mean((y_test_MeV - mass_excess_MeV)**2)
mse = np.mean((y_test_normal - mass_excess_pred)**2)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print(f"σ_RMS (MeV) : {rmse} MeV")
print(f"σ_RMS (keV) : {rmse*1000} keV")

result = pd.DataFrame({
  'Mass Excess from Data (MeV)' : y_test_normal,
  'Predicted Mass Excess (MeV)' : mass_excess_pred,
  'Absolute Error (MeV)' : np.abs(y_test_normal - mass_excess_pred)
}
)

result.to_excel('C:/Users/jovan/Downloads/NN/prediksi.xlsx', index=False)

model.save('model2input.keras')