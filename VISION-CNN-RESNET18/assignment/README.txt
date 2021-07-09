İçindekiler
Assignment3.py
TrainTestSplit.py
ResizeImage.py
ResizeHelper.py

Öncelikle ResizeImage çalıştırılır. ResizeImage DatasetResized diye bir klasör oluşturur ve içine datasetteki tüm imagelerin
(128, 128) boyutundaki hallerini kaydeder.

Daha sonra bu DatasetResized'da train-test split yapılır. TrainTestSplit çalıştırılarak.

Son olarak Assignment3'te tüm ResNet18 ile ilgili bölümler yoruma atılıp kendi modelim test edilebilir.
Kendi modelim yoruma atılarak, ResNet18'de ise FC için sadece FC kısmı açık kalıp, Conv-FC için alttaki FC yoruma atılıp 
çalıştırılabilir. 

Tüm modellerde Train ve Test için birer fonksiyon kullanılmıştır.

