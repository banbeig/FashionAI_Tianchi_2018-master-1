import gc
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import multi_gpu_model
from keras.applications.inception_resnet_v2 import preprocess_input
import inception_v4
from dataset import *
from config import *

# load data into memory
def getX():
    X = np.zeros((n, width, width, 3), dtype=np.uint8)
    for i in tqdm(range(n)):
        img = cv2.resize(cv2.imread(TRAIN_IMG_DIR+'{0}'.format(df['filename'][i])), (width, width))
        X[i] = img[:, :, ::-1]
    return X

# calculate the accuracy on validation set
def acc(y_true, y_pred):
    index = tf.reduce_any(y_true > 0.5, axis=-1)
    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    index = tf.cast(index, tf.float32)
    res = tf.cast(res, tf.float32)
    return tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)

def train(task):
    if (task == 'design'):
        task_list = task_list_design
    else:
        task_list = task_list_length
    #获取小类别名称  4种
    label_names = list(task_list.keys())
    print(n)
    #生成n行k列的全名数组，   k为每个小类别的种类
    y = [np.zeros((n, task_list[x])) for x in task_list.keys()]
    #进行one-hot编码
    for i in range(n):
        label_name = df.label_name[i]
        label = df.label[i]
        y[label_names.index(label_name)][i, label.find('y')] = 1
    #读取图像，并转换为480*480*3  rgb格式
    X = getX()
    #划分训练集 验证集  比例9:1
    n_train = int(n * 0.9)
    X_train = X[:n_train]
    X_valid = X[n_train:]
    y_train = [x[:n_train] for x in y]
    y_valid = [x[n_train:] for x in y]
    #使用生成器读入bacth数据  节省内存
    gen_train = Generator(X_train, y_train, batch_size=40, aug=True)
    #实例化inception v4模型
    base_model = inception_v4.create_model(weights='imagenet', width=width, include_top=False)
    #定义输入张量
    input_tensor = Input((width, width, 3))
    x = input_tensor
    #使用resnet v2框架，处理数据符合tf格式
    x = Lambda(preprocess_input, name='preprocessing')(x)
    #v4架构
    x = base_model(x)
    #平均池化
    x = GlobalAveragePooling2D()(x)
    #Dropout随机废弃0.5的神经元  跟池化一样 防止过拟合
    x = Dropout(0.5)(x)
    #softmax  分类count种
    x = [Dense(count, activation='softmax', name=name)(x) for name, count in task_list.items()]
    #
    model = Model(input_tensor, x)
    # model.load_weights('models/base.h5',by_name=True)
    #compile编译
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # model2 = multi_gpu_model(model, 1)

    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=[acc])
    model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=3, validation_data=(X_valid, y_valid))

    model.compile(optimizer=Adam(0.000025), loss='categorical_crossentropy', metrics=[acc])
    model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=2, validation_data=(X_valid, y_valid))

    model.compile(optimizer=Adam(0.00000625), loss='categorical_crossentropy', metrics=[acc])
    model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=3, validation_data=(X_valid, y_valid))

    model.compile(optimizer=Adam(0.00000425), loss='categorical_crossentropy', metrics=[acc])
    model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=1, validation_data=(X_valid, y_valid))

    model.compile(optimizer=Adam(0.000001), loss='categorical_crossentropy', metrics=[acc])
    model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=1, validation_data=(X_valid, y_valid))
    model.save_weights('models/%s.h5' % model_name)

    del X
    del model
    gc.collect()

# load the label file and split it into two portions
def csv_loader():
    df_test = pd.read_csv(TRAIN_LABEL_DIR, header=None)
    df_test.columns = ['filename', 'label_name', 'label']

    df_test_length = df_test[(df_test.label_name == 'skirt_length_labels') | (df_test.label_name == 'sleeve_length_labels')
                          |(df_test.label_name == 'coat_length_labels')|(df_test.label_name == 'pant_length_labels')]

    df_test_design = df_test[(df_test.label_name == 'collar_design_labels') | (df_test.label_name == 'lapel_design_labels')
                          | (df_test.label_name == 'neckline_design_labels') | (df_test.label_name == 'neck_design_labels')]
    df_test_length.to_csv(TRAIN_LENGTH_LABEL_DIR, index=False, header=None)
    df_test_design.to_csv(TRAIN_DESIGN_LABEL_DIR, index=False, header=None)


if __name__ == "__main__":
    #训练标签读取，并分为length和design两个csv文件
    csv_loader()
    #训练length，读取length标注文件，用df保存
    df = pd.read_csv(TRAIN_LENGTH_LABEL_DIR, header=None)
    #分为 图像路径/小类别名称/标注
    df.columns = ['filename', 'label_name', 'label']
    #图像打乱
    df = df.sample(frac=1).reset_index(drop=True)
    #将小类别名称后面的_label去掉
    df.label_name = df.label_name.str.replace('_labels', '')
    #n为length总数量
    n = len(df)
    train('length')
    #内存回收
    del df,n

    df = pd.read_csv(TRAIN_DESIGN_LABEL_DIR, header=None)
    df.columns = ['filename', 'label_name', 'label']
    df = df.sample(frac=1).reset_index(drop=True)
    df.label_name = df.label_name.str.replace('_labels', '')
    n = len(df)
    train('length')
