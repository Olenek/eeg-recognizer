import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

npzfile = np.load('dataset.npz')

train_x = npzfile['train_x']
train_valence, train_arousal = npzfile['train_y'].T

val_x = npzfile['val_x']
val_valence, val_arousal = npzfile['val_y'].T

reg_v = GradientBoostingRegressor(random_state=0, n_iter_no_change=10, max_features='sqrt')

reg_v.fit(train_x, train_valence)
preds_v = reg_v.predict(val_x)

preds_v_t = reg_v.predict(train_x)

mse = mean_squared_error(y_true=val_valence, y_pred=preds_v)
r2 = r2_score(y_true=val_valence, y_pred=preds_v)

print(f'VALENCE:\n\tMSE: {np.round(mse, 4)} \n\tR2: {np.round(r2, 4)}\n----------------------------------')

mse = mean_squared_error(y_true=train_valence, y_pred=preds_v_t)
r2 = r2_score(y_true=train_valence, y_pred=preds_v_t)

print(f'VALENCE:\n\tMSE: {np.round(mse, 4)} \n\tR2: {np.round(r2, 4)}\n----------------------------------')

reg_a = GradientBoostingRegressor(random_state=0, n_iter_no_change=10, max_features='sqrt')

reg_a.fit(train_x, train_arousal)
preds_a = reg_v.predict(val_x)

mse = mean_squared_error(y_true=val_arousal, y_pred=preds_a)
r2 = r2_score(y_true=val_arousal, y_pred=preds_a)
print(f'VALENCE:\n\tMSE: {np.round(mse, 4)} \n\tR2: {np.round(r2, 4)}\n----------------------------------')
