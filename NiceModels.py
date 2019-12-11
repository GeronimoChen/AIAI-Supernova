def UltraDenseResNeuro(NameShell='26_3',CellNumber=7,X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test,usegpu=True):
    # When you set cell number is zero, there is still one cell, be careful. 
    INput=Input(shape=(X_train.shape[1],1,))
    conv1=Conv1D(8,15,strides=2,padding='same')(INput)
    conv1=Conv1D(16,3,strides=1,padding='same')(conv1)
    batc1=BatchNormalization()(conv1)
    acti1=Activation('relu')(batc1)
    pool1=MaxPooling1D(2)(acti1)
    
    conv2=Conv1D(8,1)(pool1)
    batc2=BatchNormalization()(conv2)
    acti2=Activation('relu')(batc2)
    conv3=Conv1D(16,3,padding='same')(acti2)
    
    adds=[pool1]
    
    addi=Add()(adds+[conv3])
    adds.append(addi)
    
    for i in range(CellNumber):
        conv2=Conv1D(8,1)(addi)
        batc2=BatchNormalization()(conv2)
        acti2=Activation('relu')(batc2)
        conv3=Conv1D(16,3,padding='same')(acti2)
        addi=Add()(adds+[conv3])
        adds.append(addi)
    
    batc2=BatchNormalization()(addi)
    
    flat1=keras.layers.Flatten()(batc2)
    drop1=Dropout(0.2)(flat1)
    dens1=Dense(256,activation='relu')(drop1)
    drop2=Dropout(0.2)(dens1)
    dens2=Dense(128,activation='relu')(drop2)
    dens3=Dense(1,activation='sigmoid')(dens2)
    
    model=Model(inputs=INput,outputs=dens3)
    print(model.summary())
    if usegpu==True:model=keras.utils.multi_gpu_model(model,gpus=2)
    opt=keras.optimizers.adam(lr=0.00003,decay=1e-6)
    model.compile(optimizer=opt,loss='mse')
    print(model.summary())
    
    history1=model.fit(X_train,Y_train[NameShell],epochs=700,\
              validation_data=[X_test,Y_test[NameShell]],batch_size=4000,\
              callbacks=[keras.callbacks.EarlyStopping(patience=10)],verbose=2)
    opt=keras.optimizers.adam(lr=0.0000003,decay=1e-6)
    model.compile(optimizer=opt,loss='mse')
    history2=model.fit(X_train,Y_train[NameShell],epochs=700,\
              validation_data=[X_test,Y_test[NameShell]],batch_size=10000,\
              callbacks=[keras.callbacks.EarlyStopping(patience=5)],verbose=2)
    return model,history1,history2

def ConNeuro(NameShell='26_3',CellNumber=7,X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test):
    INput=Input(shape=(X_train.shape[1],1,))
    conv1=Conv1D(8,15,strides=2,padding='same')(INput)
    conv1=Conv1D(16,3,strides=1,padding='same')(conv1)
    batc1=BatchNormalization()(conv1)
    acti1=Activation('relu')(batc1)
    pool1=MaxPooling1D(2)(acti1)
    
    conv2=Conv1D(8,1)(pool1)
    batc2=BatchNormalization()(conv2)
    acti2=Activation('relu')(batc2)
    conv3=Conv1D(16,3,padding='same')(acti2)
    
    for i in range(CellNumber):
        conv2=Conv1D(8,1)(conv3)
        batc2=BatchNormalization()(conv2)
        acti2=Activation('relu')(batc2)
        conv3=Conv1D(16,3,padding='same')(acti2)
        
    batc2=BatchNormalization()(conv3)
    
    flat1=keras.layers.Flatten()(batc2)
    drop1=Dropout(0.2)(flat1)
    dens1=Dense(256,activation='relu')(drop1)
    drop2=Dropout(0.2)(dens1)
    dens2=Dense(128,activation='relu')(drop2)
    dens3=Dense(1,activation='sigmoid')(dens2)
    
    model=Model(inputs=INput,outputs=dens3)
    print(model.summary())
    model=keras.utils.multi_gpu_model(model,gpus=2)
    opt=keras.optimizers.adam(lr=0.00003,decay=1e-6)
    model.compile(optimizer=opt,loss='mse')
    print(model.summary())
    
    history1=model.fit(X_train,Y_train[NameShell],epochs=700,\
              validation_data=[X_test,Y_test[NameShell]],batch_size=4000,\
              callbacks=[keras.callbacks.EarlyStopping(patience=10)],verbose=2)
    opt=keras.optimizers.adam(lr=0.0000003,decay=1e-6)
    model.compile(optimizer=opt,loss='mse')
    history2=model.fit(X_train,Y_train[NameShell],epochs=700,\
              validation_data=[X_test,Y_test[NameShell]],batch_size=10000,\
              callbacks=[keras.callbacks.EarlyStopping(patience=5)],verbose=2)
    
    return model,history1,history2

def WiderConNeuro(NameShell='26_3',CellNumber=7,X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test):
    INput=Input(shape=(X_train.shape[1],1,))
    conv1=Conv1D(8,15,strides=2,padding='same')(INput)
    conv1=Conv1D(16,3,strides=1,padding='same')(conv1)
    batc1=BatchNormalization()(conv1)
    acti1=Activation('relu')(batc1)
    pool1=MaxPooling1D(2)(acti1)
    
    conv2=Conv1D(16,3)(pool1)
    batc2=BatchNormalization()(conv2)
    acti2=Activation('relu')(batc2)
    conv3=Conv1D(16,3,padding='same')(acti2)
    
    for i in range(CellNumber):
        conv2=Conv1D(16,3)(conv3)
        batc2=BatchNormalization()(conv2)
        acti2=Activation('relu')(batc2)
        conv3=Conv1D(16,3,padding='same')(acti2)
        
    batc2=BatchNormalization()(conv3)
    
    flat1=keras.layers.Flatten()(batc2)
    drop1=Dropout(0.2)(flat1)
    dens1=Dense(256,activation='relu')(drop1)
    drop2=Dropout(0.2)(dens1)
    dens2=Dense(128,activation='relu')(drop2)
    dens3=Dense(1,activation='sigmoid')(dens2)
    
    model=Model(inputs=INput,outputs=dens3)
    print(model.summary())
    model=keras.utils.multi_gpu_model(model,gpus=2)
    opt=keras.optimizers.adam(lr=0.00003,decay=1e-6)
    model.compile(optimizer=opt,loss='mse')
    print(model.summary())
    
    history1=model.fit(X_train,Y_train[NameShell],epochs=700,\
              validation_data=[X_test,Y_test[NameShell]],batch_size=4000,\
              callbacks=[keras.callbacks.EarlyStopping(patience=10)],verbose=2)
    opt=keras.optimizers.adam(lr=0.0000003,decay=1e-6)
    model.compile(optimizer=opt,loss='mse')
    history2=model.fit(X_train,Y_train[NameShell],epochs=700,\
              validation_data=[X_test,Y_test[NameShell]],batch_size=10000,\
              callbacks=[keras.callbacks.EarlyStopping(patience=5)],verbose=2)
    
    return model,history1,history2

def UltraDenseConcatNeuro(NameShell='26_3',CellNumber=7,\
                          X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test,usegpu=True):
    INput=Input(shape=(X_train.shape[1],1,))
    conv1=Conv1D(8,15,strides=2,padding='same')(INput)
    conv1=Conv1D(16,3,strides=1,padding='same')(conv1)
    batc1=BatchNormalization()(conv1)
    acti1=Activation('relu')(batc1)
    pool1=MaxPooling1D(2)(acti1)
    
    conv2=Conv1D(8,1)(pool1)
    batc2=BatchNormalization()(conv2)
    acti2=Activation('relu')(batc2)
    conv3=Conv1D(16,3,padding='same')(acti2)
        
    adds=[pool1]
    addi=keras.layers.Concatenate()(adds+[conv3])
    adds.append(addi)
    
    for i in range(CellNumber):
        conv2=Conv1D(8,1)(addi)
        batc2=BatchNormalization()(conv2)
        acti2=Activation('relu')(batc2)
        conv3=Conv1D(16,3,padding='same')(acti2)
        addi=keras.layers.Concatenate()(adds+[conv3])
        adds.append(addi)
    
    batc2=BatchNormalization()(addi)
    
    flat1=keras.layers.Flatten()(batc2)
    drop1=Dropout(0.2)(flat1)
    dens1=Dense(256,activation='relu')(drop1)
    drop2=Dropout(0.2)(dens1)
    dens2=Dense(128,activation='relu')(drop2)
    dens3=Dense(1,activation='sigmoid')(dens2)
    
    model=Model(inputs=INput,outputs=dens3)
    print(model.summary())
    if usegpu==True:model=keras.utils.multi_gpu_model(model,gpus=2)
    opt=keras.optimizers.adam(lr=0.00003,decay=1e-6)
    model.compile(optimizer=opt,loss='mse')
    print(model.summary())
    
    history1=model.fit(X_train,Y_train[NameShell],epochs=700,\
              validation_data=[X_test,Y_test[NameShell]],batch_size=4000,\
              callbacks=[keras.callbacks.EarlyStopping(patience=10)],verbose=2)
    opt=keras.optimizers.adam(lr=0.0000003,decay=1e-6)
    model.compile(optimizer=opt,loss='mse')
    history2=model.fit(X_train,Y_train[NameShell],epochs=700,\
              validation_data=[X_test,Y_test[NameShell]],batch_size=10000,\
              callbacks=[keras.callbacks.EarlyStopping(patience=5)],verbose=2)
    
    return model,history1,history2