# Face Detector

Hello! This project is a intented to be a final project for 2D Image Processing Course.

## Build

Let's start from the basics. You need to build the project in order to try it out yourself.

A few prerequisites should be met beforehand:
- C++ 17 compiler installed, check out [this guide](https://en.cppreference.com/w/cpp/compiler_support/17) to verify your compiler;
- OpenCV installed and configured to work with CMake: if not, please, [check out this guide](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html);
- You have stable Internet connection as [CMake needs to download dlib library](https://github.com/st235/HSE.2DImageProcessing.Week7/blob/main/Project/CMakeLists.txt#L25)

If everything is ready to go I am happy to proceed to building commands. You need to navigate to the [`Project`](https://github.com/st235/HSE.2DImageProcessing.Week7/tree/main/Project) folder and the following commands:

```bash
mkdir build
cd build

cmake ..
make -j7
```

Once is done, you should see `bin` directory inside your `build` folder. Please, do verify that the `bin` folder has an executable file and a few additional files: `dlib_face_recognition_resnet_model_v1.dat`, `haarcascade_frontalface_alt.xml`, `haarcascade_frontalface_alt2.xml`, `haarcascade_frontalface_default.xml`, `haarcascade_lefteye_2splits.xml`, `haarcascade_lefteye_2splits.xml`, and `shape_predictor_68_face_landmarks.dat`. If for some reason the files are not there you need to copy them in `bin` directory from [`misc`](https://github.com/st235/HSE.2DImageProcessing.Week7/tree/main/Project/misc).

Hooray 🎉 You're ready to start using the app.

## Data

I will breifly introduce the data before showing how to call the app through your CLI. 

I gathered photos of 10 actors for further classification:

- [Rowan Atkinson](https://en.wikipedia.org/wiki/Rowan_Atkinson)
- [Emilia Clarke](https://en.wikipedia.org/wiki/Emilia_Clarke)
- [Sacha Baron Cohen](https://en.wikipedia.org/wiki/Sacha_Baron_Cohen)
- [Benedict Cumberbatch](https://en.wikipedia.org/wiki/Benedict_Cumberbatch)
- [Martin Freeman](https://en.wikipedia.org/wiki/Martin_Freeman)
- [Nick Frost](https://en.wikipedia.org/wiki/Nick_Frost)
- [Keira Knightley](https://en.wikipedia.org/wiki/Keira_Knightley)
- [Hugh Laurie](https://en.wikipedia.org/wiki/Hugh_Laurie)
- [Andrew Lincoln](https://en.wikipedia.org/wiki/Andrew_Lincoln)
- [Simon Pegg](https://en.wikipedia.org/wiki/Simon_Pegg)

You can find gathered images inside [`Samples/Training`](https://github.com/st235/HSE.2DImageProcessing.Week7/tree/main/Samples/Training). I use these data to train my models.

To test the quality of the approach I am using annotated video samples. The videos capture the same actores as given below and contain some other people. Moreover, a few videos under the `unknown` folder predominantly contain people from the outside of the training set. You can find data gathered for testing under [`Samples/Test`](https://github.com/st235/HSE.2DImageProcessing.Week7/tree/main/Samples/Test) and follows principles from [Open Set Face Recognition](https://jwdai.github.io/Research/OSFR/OSFR.htm).

P.S.: All gathered data are used for research purposes only and towards receiving a degree. I believe that this case qualifies as [fair usage](https://www.bl.uk/business-and-ip-centre/articles/fair-dealing-copyright-explained) and obey the UK copyright law. If you have any concerns, please, contact me using my email or opening the issue in the repository.

## Preparing dataset

I made data pre-processing pior training the model.

Pre-processing can be separated into a few steps:
| Step | Image  | Description  |
| ------- | --- | --- |
| 0. Scan | ![Original](./Resources/original_face.png) | Scan the image from the `Samples` folder. |
| 1. Greyscale |  | Convert the image to greyscale to reduce amount of information. |
| 2. Find faces | ![Face](./Resources/preprocessing_face.png) | Extract faces from the image using any face detector. |
| 3. Detect eyes | ![Eyes](./Resources/preprocessing_face_with_eyes.png) | If applicable, detect eyes on within found faces and rotate them using a sample idea: left and right eyes' centers should be approximately on the same line. |
| 4. Save | ![Final](./Resources/final_face.jpg) | Save extracted faces to your disk. |
| 5. Validate |  | Discard invalid faces: bad image quality, extreme rotation angle, and so on. |


You can find pre-processed data under [`TrainSet`](https://github.com/st235/HSE.2DImageProcessing.Week7/tree/main/TrainSet) folder.

This step can be performed automagically 🪄 by using the command below:

```bash
./bin/FaceDetector ../../Samples/Training/atkinson --dataset -o ./preprocessing_results -d
```

The command accepts a list of image files and/or folder containing image files as the first parameter immediately followed by `--dataset` flag that specifies the mode. Although this arguments are making a complete command you can also find useful a few extra flags:

|Argument|Desciption|
|-----|-----|
|-d| *Debug flag*: if specified then the app displays detected face on the image. |
|-o| *Output folder*: specifies the final directory for output images. All images will be named in the following order `\*original file name\*_face_\*id of a face\*. |

### Face rotation explained

![Original](./Resources/face_rotation_explained.png)

To rotate the image back to "normal" position we need to find an angle between the x-axis and the line connecting two eyes' centers. A bit of school math and the angle can be calculated as:

```math
angle=arctan(\frac{dx}{dy})
```

You can find the correspoding code in [OpenCVFaceDetectionModel](https://github.com/st235/HSE.2DImageProcessing.Week7/blob/main/Project/src/opencv_face_detection_model.cpp#L81).

### A breif implementation overview

## Training

## Processing videos

## Quality
