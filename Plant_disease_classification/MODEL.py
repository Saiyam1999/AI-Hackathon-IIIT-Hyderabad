!pip install -U git+https://github.com/qubvel/efficientnet
import efficientnet.keras as efn

def go_for_it():                    #for initializing the base_model - EfificientNetB0
  
  model = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(440,440,3))
  return model



import osi                          #chechking and prining the harware accelerators available - Tensor Processing Unit
import tensorflow as tf
import pprint # for pretty printing our device stats

if 'COLAB_TPU_ADDR' not in os.environ:
    print('ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!')
else:
    tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    print ('TPU address is', tpu_address)

    with tf.compat.v1.Session(tpu_address) as session:
      devices = session.list_devices()

    print('TPU devices:')
    pprint.pprint(devices)




resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])          #Creating stretergy for using TPU
tf.config.experimental_connect_to_host(resolver.master())
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)



import pandas as pd
true_label=[]
df=pd.read_csv('/content/drive/My Drive/train.csv')
for idx,name in enumerate(filenames):                                           #making sure that all images have correct labels
  true_label.append(df[df['image_id']==name.split('.')[0]].iloc[0]['healthy'])
from keras.utils import to_categorical
encoded = to_categorical(true_label)


from keras.models import Sequential                                     #the model after efficient net B0
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten
from keras.optimizers import SGD

tf.compat.v1.disable_eager_execution()
with strategy.scope():
  model12=Sequential([go_for_it()])
  
  print(model12.output_shape)
  model12.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
  model12.add(Conv2D(filters=1024, kernel_size=(2,2), strides=(1,1)))
  model12.add(Conv2D(filters=256, kernel_size=(2,2), strides=(1,1)))
  model12.add(Flatten())
  print(model12.output_shape)
  model12.add(Dense(512, activation='relu'))
  model12.add(Dense(64,activation='relu'))
  model12.add(Dense(2,activation='softmax'))
  print(model12.output_shape)
  model12.compile(optimizer=SGD(learning_rate=0.0003),loss='binary_crossentropy',	metrics=['accuracy'])


model12.fit(np.array(train_data[0:1800]),  encoded[0:1800], epochs=2, batch_size=30)            #fitting the data
