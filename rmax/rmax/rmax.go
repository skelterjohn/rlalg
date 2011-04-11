package main

import (
	"fmt"
	"gonicetrace.googlecode.com/hg/nicetrace"
	"goargcfg.googlecode.com/hg/argcfg"
	"go-glue.googlecode.com/hg/rlglue"
	"github.com/skelterjohn/rlalg/rmax"
)

type Config struct {
	M	int
	Epsilon	float64
}

func main() {
	defer nicetrace.Print()
	var config rmax.RmaxConfig
	config.M = 5
	config.Epsilon = .1
	argcfg.LoadArgs(&config)
	ragent := rmax.NewRmaxAgent(config)
	if err := rlglue.LoadAgent(ragent); err != nil {
		fmt.Println("Error running rmax: %v\n", err)
	}
}
