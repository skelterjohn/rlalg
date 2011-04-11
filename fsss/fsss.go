package fsss

import (
	"fmt"
	"math"
	"gohash.googlecode.com/hg/hashlessmap"
	"go-glue.googlecode.com/hg/rltools/discrete"
)

type Config struct {
	Memoize              bool
	Shallow              bool
	C                    uint64
	MaxVisits            uint64
	EarlyTermination     bool
	UseUncertaintyRate   bool
	ZeroAtDepthThreshold bool
}

func ConfigDefault() (cfg Config) {
	cfg.Memoize = true
	cfg.Shallow = false
	cfg.C = 5
	cfg.MaxVisits = 0
	cfg.EarlyTermination = false
	cfg.UseUncertaintyRate = false
	cfg.ZeroAtDepthThreshold = false
	return
}

var Awake bool

type ActionFilter interface {
	ActionAvailable(action discrete.Action) bool
}

type Searcher struct {
	NodeDepthMaps map[uint64]*hashlessmap.Map
	Cfg           Config
	NumActions    uint64
	Vmin, Vmax    float64
	Gamma         float64
	lastPathScore float64

	Dump bool
}

func New() (s *Searcher) {
	s = &Searcher{}
	s.NodeDepthMaps = make(map[uint64]*hashlessmap.Map)
	return
}

func (s *Searcher) GetNode(depth uint64, o discrete.Oracle) (res *Node) {
	if o == nil {
		panic("nil oracle")
	}
	if !s.Cfg.Memoize {
		return newNode(s, o)
	}
	if s.Cfg.Shallow {
		depth = 0
	}
	//fmt.Printf("+*Searcher.GetNode(%v,%v)\n", depth, o)
	//defer println("-*Searcher.GetNode")
	hmap, ok := s.NodeDepthMaps[depth]
	if ok {
		ri, ok := hmap.Get(o)
		if ok {
			res = ri.(*Node)
			return
		}
	} else {
		hmap = hashlessmap.New()
		s.NodeDepthMaps[depth] = hmap
	}
	res = newNode(s, o)
	res.depth = depth
	//fmt.Fprintf(os.Stderr, "%v %v\n", o, res)
	hmap.Put(o, res)

	return
}

func (s *Searcher) ClearLevel(depth uint64) {
	//println("+*Searcher.ClearLevel")
	//defer println("-*Searcher.ClearLevel")
	s.NodeDepthMaps[depth] = nil, false
}
func (s *Searcher) RunTrajectory(n *Node, length uint64) (expanded uint64) {
	expanded, s.lastPathScore = s.RunTrajectoryAux(n, length, 0, 0)
	return
}
func (s *Searcher) RunTrajectoryNotify(n *Node, length uint64, notify chan bool) {
	s.RunTrajectoryAux(n, length, 0, 0)
	notify <- true
}

func (s *Searcher) RunTrajectoryAux(n *Node, length, depth uint64, pathScore float64) (expanded uint64, endPathScore float64) {
	//fmt.Printf("+*Searcher.RunTrajectory\n")
	//defer println("-*Searcher.RunTrajectory")

	endPathScore = pathScore

	//fmt.Printf("%v\n\n", n.o)

	if n == nil {
		panic("RunTrajectory(nil)")
	}

	if s.Dump {
		fmt.Printf("%v\n", n)
	}

	if n.terminal {
		//print("T")
		return
	}
	if length == 0 {
		if s.Cfg.ZeroAtDepthThreshold {
			n.vupper = 0
			n.vlower = 0
		}
		//print("D")
		return
	}
	if s.Cfg.MaxVisits != 0 && s.Cfg.MaxVisits <= n.visits {
		//print("M")
		return
	}

	if s.Cfg.EarlyTermination && pathScore < s.lastPathScore {
		//print("E")
		return
	}

	n.visits++

	if n.leaf {
		if !n.expand() {
			return
		} else {
			if s.Dump {
				fmt.Printf("expanded\n")
			}
			expanded += s.Cfg.C * s.NumActions
		}
	}

	a := n.getBestAction()

	if s.Dump {
		fmt.Printf("%d %f\n\n", a, n.r[a])
	}

	nn := n.getMostUncertain(a)

	stepScore := math.Log(s.Gamma) + math.Log(n.branches[a][nn]) - math.Log(float64(s.Cfg.C))

	var tailExpanded uint64
	tailExpanded, endPathScore = s.RunTrajectoryAux(nn, length-1, depth+1, pathScore+stepScore)
	expanded += tailExpanded
	n.backup()

	if s.Dump {
		fmt.Printf("backup\n%d %f\n%v\n\n", a, n.r[a], n)
	}

	return
}


func (s *Searcher) GetAction(n *Node) discrete.Action {
	return n.getBestAction()
}

func (s *Searcher) GetQs(n *Node) []float64 {
	return n.qupper
}
