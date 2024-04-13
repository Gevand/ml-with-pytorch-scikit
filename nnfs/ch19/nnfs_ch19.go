package nnfs

import (
	"fmt"
	u "nnfs/utilities"
)

func Run1() {
	fmt.Println("Fashion data set")
	u.Init()
	X, y := u.Create_fashion_data(true)
	fmt.Println(X[0])
	fmt.Println(y.At(0, 0))
}
