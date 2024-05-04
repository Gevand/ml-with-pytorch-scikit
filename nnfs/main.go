package main

import (
	"math/rand"
	nnfsch10 "nnfs/ch10"
	nnfsch14 "nnfs/ch14"
	nnfsch15 "nnfs/ch15"
	nnfsch16 "nnfs/ch16"
	nnfsch17 "nnfs/ch17"
	nnfsch18 "nnfs/ch18"
	nnfsch19 "nnfs/ch19"
	nnfsCh2_4 "nnfs/ch2-4"
	nnfsch20 "nnfs/ch20"
	nnfsCh5 "nnfs/ch5"
	nnfsCh6 "nnfs/ch6"
	nnfsCh7 "nnfs/ch7"
	nnfsch9 "nnfs/ch9"
)

func main() {
	rand.Seed(42)
	ch20()
}

func ch2_4() {
	nnfsCh2_4.Run1()
	nnfsCh2_4.Run2()
	nnfsCh2_4.Run3()
	nnfsCh2_4.Run4()
	nnfsCh2_4.Run5()
	nnfsCh2_4.Run6()
	nnfsCh2_4.Run7()
	nnfsCh2_4.Run8()
	nnfsCh2_4.Run9()
	nnfsCh2_4.Run10()
}
func ch5() {
	nnfsCh5.Run1()
	nnfsCh5.Run2()
	nnfsCh5.Run3()
	nnfsCh5.Run4()
	nnfsCh5.Run5()
	nnfsCh5.Run6()
	nnfsCh5.Run7()
}
func ch6() {
	nnfsCh6.Run1()
	nnfsCh6.Run2()
	nnfsCh6.Run3()
	nnfsCh6.Run4()
}

func ch7() {
	nnfsCh7.Run1()
	nnfsCh7.Run2()
	nnfsCh7.Run3()
	nnfsCh7.Run4()
}

func ch8() {
	//NOTHING - pure math chapter
}

func ch9() {
	nnfsch9.Run1()
	nnfsch9.Run2()
	nnfsch9.Run3()
	nnfsch9.Run4()
	nnfsch9.Run5()
	nnfsch9.Run6()
	nnfsch9.Run7()
	nnfsch9.Run8()
	nnfsch9.Run9()
	nnfsch9.Run10()
	nnfsch9.Run11()
	nnfsch9.Run12()
}
func ch10() {
	//keep these commented out, they take a bit longer and make imgages, just uncomment whichever one you want to test it
	//nnfsch10.Run1()
	//nnfsch10.Run2()
	//nnfsch10.Run3()
	//nnfsch10.Run4()
	nnfsch10.Run5()
}

func ch14() {
	nnfsch14.Run1()
	nnfsch14.Run2()
	nnfsch14.Run3()
}

func ch15() {
	nnfsch15.Run1()
	nnfsch15.Run2()
	nnfsch15.Run3()
}

func ch16() {
	nnfsch16.Run1()
}

func ch17() {
	nnfsch17.Run1()
	nnfsch17.Run2()
}

func ch18() {
	nnfsch18.Run1()
}

func ch19() {
	//nnfsch19.Run1()
	//nnfsch19.Run2()
	nnfsch19.Run3()
}

func ch20() {
	nnfsch20.Run1()
	nnfsch20.Run2()
}
