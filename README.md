# Face Detector

## Possible commands

Generate dataset

```bash
./bin/FaceDetector ../../Samples/Training/pegg --ds -f ../../Samples/Training/haarcascade_frontalface_alt2.xml -re ../../Samples/Training/haarcascade_righteye_2splits.xml -le ../../Samples/Training/haarcascade_lefteye_2splits.xml -o ../../Dataset/pegg
```

Train model

```bash
./bin/FaceDetector ../../Dataset --train -f ../../Samples/Training/haarcascade_frontalface_alt2.xml -re ../../Samples/Training/haarcascade_righteye_2splits.xml -le ../../Samples/Training/haarcascade_lefteye_2splits.xml -om ../../Dataset/bow_svm_model.yml -ol ../../Dataset/labels_mapping.dat
```
