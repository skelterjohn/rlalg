package main

import (
	"fmt"
	"gonicetrace.googlecode.com/hg/nicetrace"
	"goargcfg.googlecode.com/hg/argcfg"
	"go-glue.googlecode.com/hg/rlglue"
	"go-glue.googlecode.com/hg/rltools/discrete"
	"github.com/skelterjohn/rlalg/beb"
)

type Config struct {
	BEB	beb.BebConfig
	Which	int
}

func getRewardFooPaint(task *rlglue.TaskSpec) (foo beb.RewardFunc) {
	foo = func(s discrete.State, a discrete.Action) (r float64) {
		svalues := task.Obs.Ints.Values(s.Hashcode())
		avalues := task.Act.Ints.Values(a.Hashcode())
		if avalues[1] == 3 {
			which := avalues[0]
			objvals := svalues[which*4 : (which+1)*4]
			if objvals[0] == 1 && objvals[1] == 1 && objvals[2] == 0 && objvals[3] == 0 {
				return 10
			} else {
				return -1000000
			}
		}
		return -1
	}
	return
}
func main() {
	defer nicetrace.Print()
	var config Config
	config.BEB = beb.BebConfigDefault()
	argcfg.LoadArgs(&config)
	var getRFoo func(task *rlglue.TaskSpec) (foo beb.RewardFunc)
	switch config.Which {
	case 0:
		getRFoo = func(task *rlglue.TaskSpec) (foo beb.RewardFunc) {
			foo = func(s discrete.State, a discrete.Action) (r float64) {
				return -1
			}
			return
		}
	case 2:
		getRFoo = getRewardFooPaint
	}
	bebagent := beb.NewBebAgent(config.BEB, getRFoo)
	if err := rlglue.LoadAgent(bebagent); err != nil {
		fmt.Println("Error running rmax: %v\n", err)
	}
}
