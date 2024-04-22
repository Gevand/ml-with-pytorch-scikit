package nnfs

import (
	"fmt"
	"image"
	u "nnfs/utilities"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"gonum.org/v1/gonum/mat"
)

func Run1() {
	fmt.Println("Fashion data set")
	u.Init()
	X, y := u.Create_fashion_data(true)
	fmt.Println(X[1000*7+2])
	fmt.Println(y[0])

	a := app.New()
	w := a.NewWindow("Images")

	img := canvas.NewImageFromImage(generateImage(X[1000*7+2]))
	w.SetContent(img)
	w.Resize(fyne.NewSize(640, 480))

	w.ShowAndRun()
}

func Run2() {
	fmt.Println("Data prep")
	X, y := u.Create_fashion_data(true)
	u.Shuffle(X, y)
	u.Scale_fashion_data(X)
	X_reshape := u.Reshape_fashion_data(X)
	fmt.Println(X[0], y[0])
	fmt.Println(X_reshape[0], y[0])

	EPOCHS := 10
	BATCH_SIZE := 128

	steps := len(X_reshape) / BATCH_SIZE
	if steps*BATCH_SIZE < len(X_reshape) {
		steps += 1
	}

	for epoch := 0; epoch < EPOCHS; epoch++ {
		for step := 0; step < steps; step++ {
			batch_X := X_reshape[step*BATCH_SIZE : (step+1)*BATCH_SIZE]
			batch_Y := y[step*BATCH_SIZE : (step+1)*BATCH_SIZE]

			fmt.Println(len(batch_X), len(batch_Y))
		}
	}
}
func Run3() {
	fmt.Println("Training images")
	X, y := u.Create_fashion_data(false)
	//u.Shuffle(X, y)
	u.Scale_fashion_data(X)
	X_reshape := u.Reshape_fashion_data(X)

	model := u.NewModel()
	model.Accuracy = u.NewAccuracy_Classification()
	model.Add(u.NewLayerDense(28*28, 128))
	model.Add(u.NewActivationRelu())
	model.Add(u.NewLayerDense(128, 128))
	model.Add(u.NewActivationRelu())
	model.Add(u.NewLayerDense(128, 10))
	loss_activation := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	model.Add(loss_activation)
	model.Set(loss_activation.Loss, u.NewOptimizerAdam(0.001, 1e-3, 1e-7, 0.9, 0.999))
	model.Finalize()
	model.Train_image(X_reshape[0:2], y[0:2], 20, 128, 100)
}

func generateImage(dense *mat.Dense) image.Image {
	raw_data := dense.RawMatrix().Data
	rect := image.Rect(0, 0, 28, 28)
	img := image.NewGray(rect)
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			index := y*28 + x
			img.Pix[index] = uint8(raw_data[index])
		}
	}
	return img
}
