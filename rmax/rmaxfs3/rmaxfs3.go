package main

import (
	"strings"
	"strconv"
	"rand"
	"gonicetrace.googlecode.com/hg/nicetrace"
	"goargcfg.googlecode.com/hg/argcfg"
	"gostat.googlecode.com/hg/stat"
	"go-glue.googlecode.com/hg/rlglue"
	"github.com/skelterjohn/rlalg/rmax"
	"go-glue.googlecode.com/hg/rltools/discrete"
	"github.com/skelterjohn/rlalg/fsss"
)

type Config struct {
	M		uint64
	Depth		uint64
	NumTrajectories	uint64
	Budget		uint64
	FS3		fsss.Config
}
type RmaxFSSSAgent struct {
	task			*rlglue.TaskSpec
	rmdp			*rmax.RmaxMDP
	lastState		discrete.State
	lastAction		discrete.Action
	Cfg			Config
	s			*fsss.Searcher
	mdpo			*discrete.MDPOracle
	stepsWithPlanner	uint64
}

func NewRmaxFSSSAgent(cfg Config) (ra *RmaxFSSSAgent) {
	ra = new(RmaxFSSSAgent)
	ra.Cfg = cfg
	return
}
func (ra *RmaxFSSSAgent) AgentInit(taskString string) {
	ra.task, _ = rlglue.ParseTaskSpec(taskString)
	ra.rmdp = rmax.NewRmaxMDP(ra.task, ra.Cfg.M)
}
func (ra *RmaxFSSSAgent) AgentStart(obs rlglue.Observation) (act rlglue.Action) {
	ra.stepsWithPlanner = 0
	ra.lastState = discrete.State(ra.task.Obs.Ints.Index(obs.Ints()))
	ra.Plan()
	act = rlglue.NewAction(ra.task.Act.Ints.Values(ra.GetAction().Hashcode()), []float64{}, []byte{})
	ra.lastAction = discrete.Action(ra.task.Act.Ints.Index(act.Ints()))
	return
}
func (ra *RmaxFSSSAgent) AgentStep(reward float64, obs rlglue.Observation) (act rlglue.Action) {
	ra.stepsWithPlanner++
	nextState := discrete.State(ra.task.Obs.Ints.Index(obs.Ints()))
	learned := ra.rmdp.Observe(ra.lastState, ra.lastAction, nextState, reward)
	if learned {
		ra.Forget()
	}
	ra.lastState = nextState
	ra.Plan()
	act = rlglue.NewAction(ra.task.Act.Ints.Values(ra.GetAction().Hashcode()), []float64{}, []byte{})
	ra.lastAction = discrete.Action(ra.task.Act.Ints.Index(act.Ints()))
	return
}
func (ra *RmaxFSSSAgent) AgentEnd(reward float64) {
	learned := ra.rmdp.ObserveTerminal(ra.lastState, ra.lastAction, reward)
	if learned {
		ra.Forget()
	}
}
func (ra *RmaxFSSSAgent) AgentCleanup() {
}
func (ra *RmaxFSSSAgent) AgentMessage(message string) string {
	tokens := strings.Split(message, " ", -1)
	if tokens[0] == "seed" {
		seed, _ := strconv.Atoi64(tokens[1])
		rand.Seed(seed)
	}
	return ""
}
func (ra *RmaxFSSSAgent) GetAction() (action discrete.Action) {
	if ra.s == nil {
		action = discrete.Action(stat.NextRange(int64(ra.task.Act.Ints.Count())))
		return
	}
	node := ra.s.GetNode(ra.stepsWithPlanner, ra.mdpo)
	action = discrete.Action(ra.s.GetAction(node))
	ra.s.ClearLevel(ra.stepsWithPlanner)
	return
}
func (ra *RmaxFSSSAgent) Forget() {
	ra.mdpo = discrete.NewMDPOracle(ra.rmdp, ra.lastState)
	ra.s = fsss.New()
	ra.s.Cfg = ra.Cfg.FS3
	ra.s.NumActions = ra.rmdp.NumActions()
	ra.s.Gamma = ra.task.DiscountFactor
	if ra.s.Gamma == 1 {
		ra.s.Gamma = 0.9
	}
	ra.s.Vmin = ra.task.Reward.Min / (1 - ra.s.Gamma)
	ra.s.Vmax = 5
	ra.stepsWithPlanner = 0
}
func (ra *RmaxFSSSAgent) Plan() {
	if ra.s == nil {
		return
	}
	ra.mdpo = ra.mdpo.Teleport(ra.lastState)
	root := ra.s.GetNode(ra.stepsWithPlanner, ra.mdpo)
	var expanded uint64
	for i := 0; i < int(ra.Cfg.NumTrajectories); i++ {
		expanded += ra.s.RunTrajectory(root, ra.Cfg.Depth)
		if ra.Cfg.Budget != 0 && expanded > ra.Cfg.Budget {
			break
		}
	}
}
func main() {
	defer nicetrace.Print()
	var cfg Config
	cfg.M = 5
	cfg.Depth = 10
	cfg.NumTrajectories = 500
	cfg.FS3 = fsss.ConfigDefault()
	argcfg.LoadArgs(&cfg)
	agent := NewRmaxFSSSAgent(cfg)
	rlglue.LoadAgent(agent)
}
