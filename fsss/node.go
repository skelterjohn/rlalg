package fsss

import (
	"sync"
	"fmt"
	"rand"
	"go-glue.googlecode.com/hg/rltools/discrete"
)

type Node struct {
	s *Searcher
	//used for synchronization
	block sync.Mutex
	o     discrete.Oracle
	//is this Node a leaf?
	leaf bool
	//how many times each action has been attempted
	totals []float64
	//current estimate for each action's reward
	r []float64
	//current estimate for each action's value bounds
	qlower, qupper []float64
	//current estimate for Node's value bounds
	vlower, vupper float64
	//the depth at which this Node resides
	depth uint64
	//number of trajectories running through this node
	visits uint64
	//the set of nexts, along with how many times they have occured
	branches []map[*Node]float64

	//quickies
	terminal              bool
	currentBestAction     discrete.Action
	currentMostUncertains []*Node
}

func (n *Node) GetBranch(a int) map[*Node]float64 {
	return n.branches[a]
}
func (n *Node) GetR(a int) float64 {
	return n.r[a]
}

func (n *Node) String() (res string) {
	res = fmt.Sprintf("n{\no%v\n r{%v}\nvlower{%f} vupper{%f}\nqlower{%v}\nqupper{%v}\npi{%d}}", n.o, n.r, n.vlower, n.vupper, n.qlower, n.qupper, n.currentBestAction)
	return
}

func (n *Node) GetOracle() (o discrete.Oracle) {
	return n.o
}
func (n *Node) GetDepth() (depth uint64) {
	return n.depth
}
func (n *Node) GetValue() (upper, lower float64) {
	return n.vupper, n.vlower
}

func newNode(s *Searcher, o discrete.Oracle) (n *Node) {
	n = &Node{}
	n.s = s
	n.o = o
	n.leaf = true
	n.terminal = o.Terminal()
	if o.Terminal() {
		//n.vupper, n.vlower = 0, 0
	} else {
		n.totals = make([]float64, s.NumActions)
		n.r = make([]float64, s.NumActions)
		n.qlower = make([]float64, s.NumActions)
		n.qupper = make([]float64, s.NumActions)
		n.branches = make([]map[*Node]float64, s.NumActions)
		for a := uint64(0); a < s.NumActions; a++ {
			n.qlower[a] = s.Vmin
			n.qupper[a] = s.Vmax
			n.branches[a] = make(map[*Node]float64)
		}
		n.currentMostUncertains = make([]*Node, s.NumActions)
		n.vupper = s.Vmax
		n.vlower = s.Vmin
	}
	return
}

//argmax over qvalues to find the best action
func (n *Node) getBestAction() (bestAction discrete.Action) {
	n.block.Lock()
	defer n.block.Unlock()
	return n.currentBestAction
}

//argmax over the branch set to find the most uncertain next state
func (n *Node) getMostUncertain(a discrete.Action) (unn *Node) {
	n.block.Lock()
	defer n.block.Unlock()
	if n.currentMostUncertains[a] == nil {
		panic("mostUncertain is nil")
	}
	return n.currentMostUncertains[a]
}

func (n *Node) getUncertainty() (uncertainty float64) {
	uncertainty = n.vupper - n.vlower
	if n.s.Cfg.UseUncertaintyRate && n.visits != 0 {
		uncertainty = (n.s.Vmax - n.s.Vmin - uncertainty) / float64(n.visits)
	}
	return
}

var showoff = false

func (n *Node) Backup() {
	showoff = true
	n.backup()
	showoff = false
}
func (n *Node) backup() {
	n.block.Lock()
	defer n.block.Unlock()
	//fmt.Printf("+*Node.backup(%v)()\n", n.o.Hashcode())
	//defer println("-*Node.backup")
	n.currentBestAction = 0
	//max it
	n.vlower = n.s.Vmin
	n.vupper = n.s.Vmin
	//update the qs for each action

	offset := uint64(rand.Intn(int(n.s.NumActions)))

	af, haveActionFilter := n.o.(ActionFilter)

	for ao := uint64(0); ao < n.s.NumActions; ao++ {
		a := discrete.Action((ao + offset) % n.s.NumActions)
		avail := !haveActionFilter || af.ActionAvailable(a)
		n.qlower[a] = 0
		n.qupper[a] = 0
		//E[V(s')] part
		var mostUncertainty float64
		for nn, count := range n.branches[a] {
			weight := count / float64(n.s.Cfg.C)
			wvupper := weight * nn.vupper
			wvlower := weight * nn.vlower
			n.qlower[a] += wvlower
			n.qupper[a] += wvupper
			uncertainty := nn.getUncertainty() * count
			if avail && uncertainty >= mostUncertainty {
				mostUncertainty, n.currentMostUncertains[a] = uncertainty, nn
			}
		}
		//gamma part
		n.qlower[a] *= n.s.Gamma
		n.qupper[a] *= n.s.Gamma
		//R part
		n.qlower[a] += n.r[a]
		n.qupper[a] += n.r[a]
		if avail {
			//max operator
			if n.qlower[a] > n.vlower {
				n.vlower = n.qlower[a]
			}
			if n.qupper[a] > n.vupper {
				n.vupper = n.qupper[a]
				n.currentBestAction = a
			}
		}
	}
}

func (n *Node) expand() (expanded bool) {
	n.block.Lock()
	defer n.block.Unlock()
	//println("+*Node.expand")
	//defer println("-*Node.expand")
	if n.o.Terminal() {
		return false
	}
	expanded = true
	n.leaf = false

	af, haveActionFilter := n.o.(ActionFilter)

	for a := discrete.Action(0); a.Hashcode() < n.s.NumActions; a++ {
		avail := !haveActionFilter || af.ActionAvailable(a)
		n.r[a] = 0
		var mostUncertainty float64
		for i := uint64(0); i < n.s.Cfg.C; i++ {
			no, r := n.o.Next(a)
			if no == nil {
				panic("Next() -> nil")
			}
			n.r[a] += r
			//get the Node for no (next oracle)
			nn := n.s.GetNode(n.depth+1, no)
			count := n.branches[a][nn] + 1
			n.branches[a][nn] = count

			uncertainty := count * nn.getUncertainty()
			if avail && uncertainty >= mostUncertainty {
				mostUncertainty, n.currentMostUncertains[a] = uncertainty, nn
			}
		}
		n.r[a] /= float64(n.s.Cfg.C)
	}

	n.o = nil

	return
}
