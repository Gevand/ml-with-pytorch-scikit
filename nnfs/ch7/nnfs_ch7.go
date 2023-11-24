package nnfs

import (
	"fmt"
	"math"
	u "nnfs/utilities"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func Run1() {
	fmt.Println("Starting ch 7 Derivatives")

	pts1 := make(plotter.XYs, 5)

	p := plot.New()
	p.Title.Text = "Linear function y=2x graphed"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	for i := range pts1 {
		pts1[i].X = float64(i)
		pts1[i].Y = float64(2 * i)
	}
	err := plotutil.AddLinePoints(p,
		"First", pts1)
	if err != nil {
		panic(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch7/2x.png"); err != nil {
		panic(err)
	}
}

func Run2() {
	fmt.Println("Slope of 2x^2")

	pts1 := make(plotter.XYs, 5)
	slope := make(plotter.XYs, 4)

	p := plot.New()
	p.Title.Text = "Approximation of the eparabolic functions example tangents"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	for i := range pts1 {
		pts1[i].X = float64(i)
		pts1[i].Y = float64(2.0 * math.Pow(float64(i), 2))
	}
	for i := range slope {
		slope[i].X = float64(i)
		slope[i].Y = (pts1[i+1].Y - pts1[i].Y) / (pts1[i+1].X - pts1[i].X)
	}

	err := plotutil.AddLinePoints(p,
		"2x^2", pts1, "Slope", slope)
	if err != nil {
		panic(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch7/2xSquared.png"); err != nil {
		panic(err)
	}
}

func Run3() {
	fmt.Println("Numerical Derivatives")

	f := func(x float64) float64 {
		return 2 * math.Pow(x, 2)
	}

	x1 := 4.0
	x2 := 4 + 0.0000001

	y1 := f(x1)
	y2 := f(x2)

	derivative := (y2 - y1) / (x2 - x1)

	fmt.Println("Numerical method derivative is", derivative, "But really should be 16 (derivate of 2x^2 = 4x)")
}

func Run4() {
	fmt.Println("Putting everyting together")

	f := func(x float64) float64 {
		return 2 * math.Pow(x, 2)
	}

	X := u.Linspace(0, 5, 5/0.001)
	y := make([]float64, len(X))

	for i := range X {
		y[i] = f(X[i])
	}

	p2_delta := 0.0001
	x1 := 2.0
	x2 := x1 + p2_delta

	y1 := f(x1)
	y2 := f(x2)

	approximate_derivate := (y2 - y1) / (x2 - x1)
	//since y = mx + b, then b = y - mx where m is our approximate derivative

	b := y2 - approximate_derivate*x2
	tangent_line := func(x float64) float64 {
		return approximate_derivate*x + b
	}

	p := plot.New()
	p.Title.Text = "Graphed approximate derivative for f(x) where x=2"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	pts1 := make(plotter.XYs, len(X))
	for i := range pts1 {
		pts1[i].X = X[i]
		pts1[i].Y = y[i]
	}

	tangent_1 := make(plotter.XYs, 100)
	counter := -1 * 100 / 2.0
	for i := range tangent_1 {
		tangent_1[i].X = x1 + counter*0.02
		tangent_1[i].Y = tangent_line(tangent_1[i].X)
		counter = counter + 1
	}

	//I know this can be a loop, but I'm lazy
	x1 = 4.0
	x2 = x1 + p2_delta
	y1 = f(x1)
	y2 = f(x2)
	approximate_derivate = (y2 - y1) / (x2 - x1)
	b = y2 - approximate_derivate*x2

	tangent_2 := make(plotter.XYs, 100)
	counter = -1 * 100 / 2.0
	for i := range tangent_2 {
		tangent_2[i].X = x1 + counter*0.02
		tangent_2[i].Y = tangent_line(tangent_2[i].X)
		counter = counter + 1
	}

	err := plotutil.AddLines(p,
		"2x^2", pts1, "Tangent 1", tangent_1, "Tangent 2", tangent_2)
	if err != nil {
		panic(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch7/derivative.png"); err != nil {
		panic(err)
	}
}
