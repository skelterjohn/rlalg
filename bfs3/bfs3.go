package bfs3

import (
	"time"
	"os"
	"fmt"
	"gostat.googlecode.com/hg/stat"
	"go-glue.googlecode.com/hg/rlglue"
	"go-glue.googlecode.com/hg/rltools/discrete"
	"github.com/skelterjohn/rlbayes"
	"github.com/skelterjohn/rlalg/fsss"
)

type Prior func(task *rlglue.TaskSpec) bayes.BeliefState
type Config struct {
	MaxTrajectories	uint64
	Depth		uint64
	Budget		uint64
	ReplanEachStep	bool
	CustomGammaV	bool
	Gamma		float64
	Vmin, Vmax	float64
	FS3		fsss.Config
}

func ConfigDefault() (cfg Config) {
	cfg.MaxTrajectories = 500
	cfg.Depth = 10
	cfg.Budget = 1000
	cfg.ReplanEachStep = false
	cfg.CustomGammaV = false
	cfg.Gamma = 0.9
	cfg.Vmin, cfg.Vmax = 0, 1
	cfg.FS3 = fsss.ConfigDefault()
	return
}

type BFS3Agent struct {
	task			*rlglue.TaskSpec
	prior			Prior
	belief			bayes.BeliefState
	lastAction		discrete.Action
	stepsWithPlanner	uint64
	discoveries		uint64
	fs3			*fsss.Searcher
	Cfg			Config
	Counter			uint64
	Dump			bool
}

func New(prior Prior) (this *BFS3Agent) {
	this = new(BFS3Agent)
	this.prior = prior
	return
}
func (this *BFS3Agent) GetBelief() bayes.BeliefState {
	return this.belief
}
func (this *BFS3Agent) getActionIndex(act rlglue.Action) (index uint64) {
	return this.task.Act.Ints.Index(act.Ints())
}
func (this *BFS3Agent) getIndexAction(index discrete.Action) (act rlglue.Action) {
	return rlglue.NewAction(this.task.Act.Ints.Values(index.Hashcode()), []float64{}, []byte{})
}
func (this *BFS3Agent) getStateIndex(state rlglue.Observation) (index uint64) {
	return this.task.Obs.Ints.Index(state.Ints())
}
func (this *BFS3Agent) getAction() (index discrete.Action) {
	if this.fs3.Dump {
		println("getAction")
	}
	if this.Cfg.ReplanEachStep {
		this.ResetPlanner()
	}
	if this.fs3 == nil {
		index = discrete.Action(stat.NextRange(int64(this.task.Act.Ints.Count())))
		return
	}
	node := this.fs3.GetNode(this.stepsWithPlanner, this.belief)
	var expanded uint64
	for i := uint64(0); i < this.Cfg.MaxTrajectories; i++ {
		expandedThisTime := this.fs3.RunTrajectory(node, this.Cfg.Depth)
		expanded += expandedThisTime
		if this.Cfg.Budget != 0 && expanded > this.Cfg.Budget {
			break
		}
	}
	if this.Dump {
		this.fs3.Dump = true
		fmt.Printf("root:\n%v\n\n", node)
		this.fs3.RunTrajectory(node, this.Cfg.Depth)
	}
	index = discrete.Action(this.fs3.GetAction(node))
	fmt.Fprintf(os.Stderr, "%v\n", this.fs3.GetQs(node))
	if !this.Cfg.FS3.Shallow {
		this.fs3.ClearLevel(this.stepsWithPlanner)
		this.stepsWithPlanner++
	}
	time.Sleep(1e9)
	return
}
func (this *BFS3Agent) ResetPlanner() {
	this.fs3 = fsss.New()
	this.fs3.Cfg = this.Cfg.FS3
	this.fs3.NumActions = uint64(this.task.Act.Ints.Count())
	if this.Cfg.CustomGammaV {
		this.fs3.Gamma = this.Cfg.Gamma
		this.fs3.Vmin = this.Cfg.Vmin
		this.fs3.Vmax = this.Cfg.Vmax
	} else {
		this.fs3.Gamma = this.task.DiscountFactor
		if this.fs3.Gamma == 1 {
			this.fs3.Gamma = 0.95
		}
		this.fs3.Vmin = this.task.Reward.Min / (1 - this.fs3.Gamma)
		this.fs3.Vmax = this.task.Reward.Max / (1 - this.fs3.Gamma)
	}
}
func (this *BFS3Agent) AgentInit(taskString string) {
	this.task, _ = rlglue.ParseTaskSpec(taskString)
	this.belief = this.prior(this.task)
	this.ResetPlanner()
}
func (this *BFS3Agent) AgentStart(state rlglue.Observation) (act rlglue.Action) {
	if this.fs3.Dump {
		println("AgentStart")
	}
	s := discrete.State(this.getStateIndex(state))
	if s != this.belief.GetState() {
		this.belief.Teleport(s)
	}
	this.lastAction = this.getAction()
	act = this.getIndexAction(this.lastAction)
	return
}
func (this *BFS3Agent) AgentStep(reward float64, state rlglue.Observation) (act rlglue.Action) {
	s := discrete.State(this.getStateIndex(state))
	old := this.belief
	this.belief = this.belief.Update(this.lastAction, s, reward)
	if this.belief.LessThan(old) || old.LessThan(this.belief) {
		this.discoveries++
	}
	this.lastAction = this.getAction()
	act = this.getIndexAction(this.lastAction)
	return
}
func (this *BFS3Agent) AgentEnd(reward float64) {
	old := this.belief
	this.belief = this.belief.UpdateTerminal(this.lastAction, reward)
	if this.belief.LessThan(old) || old.LessThan(this.belief) {
		this.discoveries++
	}
	return
}
func (this *BFS3Agent) AgentCleanup() {
	return
}
func (this *BFS3Agent) AgentMessage(message string) (reply string) {
	return
}
