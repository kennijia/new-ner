from predict import predict

text = '当水库水位超过 163m 时，要每隔一小时将水库水位、降雨量等情况报告镇三防指挥所。'
print('TEXT:', text)
print('PRED:', predict(text))
