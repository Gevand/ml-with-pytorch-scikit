package nnfs

import (
	"fmt"
	"image"
	u "nnfs/utilities"

	"fyne.io/fyne"
	"fyne.io/fyne/app"
	"fyne.io/fyne/canvas"
	"gonum.org/v1/gonum/mat"
)

func Run1() {
	fmt.Println("Fashion data set")
	u.Init()
	X, y := u.Create_fashion_data(true)
	fmt.Println(X[0])
	fmt.Println(y.At(0, 0))

	a := app.New()
	w := a.NewWindow("Images")

	img := canvas.NewImageFromImage(generateImage(X[0]))
	w.SetContent(img)
	w.Resize(fyne.NewSize(640, 480))

	w.ShowAndRun()
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
