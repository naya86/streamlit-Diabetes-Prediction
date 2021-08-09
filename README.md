# Diabets-prediction      



## _Streamlit 당뇨병 예측 앱_


- Python 3.7
- Tensorflow   
   
      
         
## _데이터셋_
- 캐글 데이터셋 ( 768개의 데이터 )   

## _데이터 전처리_   
- X,y 로 데이터 분류
  - **X = Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age** Columns
  - **y = Outcome** Column

- Nan 데이터 처리
  - Nan 데이터 확인 -> 없음
  - 수치가 0인 데이터가 확인 됨 
    - sklearn 의 Imputer를 활용 ( 수치가 0인 데이터를 각 Column별 평균값(mean)으로 변경 후 X 값 저장
 
- y값 불균형 확인 ( Outcome값이 0 : 500개,  1 : 268개) 
  - **SMOTE를 활용 Oversampling -> 0 : 500개, 1 : 500개**
 
- 피처스케일링
  - MinMaxScaler 활용, 피처스케일링 ( 0 ~ 1의 값 )
  
- train, test set 나누기
  - skleran 의 train_test_split로 8:2 비율로 처리 ( train : test )


## _예측 모델 만들기_

- Keras Classifier 모델 
```python
def build_classifier(learning_rate = 0.01) :
  model = Sequential()
  model.add( Dense(input_dim=8, units=16, activation='relu'))
  model.add( Dense(32, activation='relu'))
  model.add( Dense( 512, activation='relu'))
  model.add( Dense(1, activation='sigmoid'))
  optimizer = tensorflow.keras.optimizers.Adam(learning_rate = learning_rate)
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics='accuracy')

  return model
```

```python
model = KerasClassifier(build_fn=build_classifier)
```

- 모델 학습 
  - Callback 함수 이용 , 모델의 Best Accuracy 값으로 모델 저장
  - 학습 
  ```python
  history = model.fit(X_train, y_train, epochs=500,  batch_size=10,validation_data=(X_test, y_test), callbacks=cp )
  ```
  

  
  
  
  
  





 






