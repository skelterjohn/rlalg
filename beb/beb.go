package beb

import (
	"go-glue.googlecode.com/hg/rlglue"
	"go-glue.googlecode.com/hg/rltools/discrete"
	"github.com/skelterjohn/rlalg/vi"
)

type RewardFunc func(s discrete.State, a discrete.Action) (r float64)
type BebConfig struct {
	Beta	float64
	Epsilon	float64
	RFoo	RewardFunc
}

func BebConfigDefault() (cfg BebConfig) {
	cfg.Beta = 1
	cfg.Epsilon = .1
	cfg.RFoo = nil
	return
}

type BebMDP struct {
	discrete.FlatMDP
	CountsSA	[][]int
	CountsSAN	[][][]int
	TotalR		[][]float64
	Beta		float64
	RFoo		RewardFunc
}

func NewBebMDP(task *rlglue.TaskSpec, Cfg BebConfig) (this *BebMDP) {
	numStates := task.Obs.Ints.Count()
	numActions := task.Act.Ints.Count()
	this = new(BebMDP)
	this.Task = task
	this.Transitions = make([][][]float64, numStates)
	this.Rewards = make([][]float64, numStates)
	this.Gamma = task.DiscountFactor
	this.CountsSA = make([][]int, numStates)
	this.CountsSAN = make([][][]int, numStates)
	this.TotalR = make([][]float64, numStates)
	for s := range this.CountsSA {
		this.CountsSA[s] = make([]int, numActions)
		this.CountsSAN[s] = make([][]int, numActions)
		for a, _ := range this.CountsSAN[s] {
			this.CountsSAN[s][a] = make([]int, numStates)
		}
		this.TotalR[s] = make([]float64, numActions)
		this.Rewards[s] = make([]float64, numActions)
		this.Transitions[s] = make([][]float64, numActions)
		for a, _ := range this.Transitions[s] {
			this.Transitions[s][a] = make([]float64, numStates)
		}
	}
	this.Beta = Cfg.Beta
	return
}
func (rm *BebMDP) T(s discrete.State, a discrete.Action, n discrete.State) float64 {
	return rm.Transitions[s][a][n]
}
func (this *BebMDP) R(s discrete.State, a discrete.Action) float64 {
	n := float64(this.CountsSA[s][a])
	bonus := this.Beta / (1 + n)
	r := this.Rewards[s][a]
	if this.RFoo != nil {
		r = this.RFoo(s, a)
	}
	return r + bonus
}
func (rm *BebMDP) resolve(s discrete.State, a discrete.Action) {
	if rm.Transitions[s] == nil {
		rm.Transitions[s] = make([][]float64, len(rm.CountsSAN[s]))
	}
	if rm.Transitions[s][a] == nil {
		rm.Transitions[s][a] = make([]float64, len(rm.CountsSAN[s][a]))
	}
	norm := 1.0 / float64(rm.CountsSA[s][a])
	for n, count := range rm.CountsSAN[s][a] {
		p := float64(count) * norm
		rm.Transitions[s][a][n] = p
	}
	rm.Rewards[s][a] = rm.TotalR[s][a] * norm
}
func (rm *BebMDP) Observe(s discrete.State, a discrete.Action, n discrete.State, r float64) (learned bool) {
	rm.CountsSA[s][a]++
	rm.CountsSAN[s][a][n]++
	rm.resolve(s, a)
	return true
}
func (rm *BebMDP) ObserveTerminal(s discrete.State, a discrete.Action, r float64) (learned bool) {
	rm.CountsSA[s][a]++
	return true
}

type BebAgent struct {
	task		*rlglue.TaskSpec
	rmdp		*BebMDP
	qt		*discrete.QTable
	lastState	discrete.State
	lastAction	discrete.Action
	Cfg		BebConfig
	GetRFoo		func(task *rlglue.TaskSpec) (foo RewardFunc)
}

func NewBebAgent(Cfg BebConfig, GetRFoo func(task *rlglue.TaskSpec) (foo RewardFunc)) (ra *BebAgent) {
	ra = new(BebAgent)
	ra.Cfg = Cfg
	ra.GetRFoo = GetRFoo
	return
}
func (ra *BebAgent) AgentInit(taskString string) {
	ra.task, _ = rlglue.ParseTaskSpec(taskString)
	if ra.task.DiscountFactor == 1 {
		ra.task.DiscountFactor = 0.99
	}
	ra.rmdp = NewBebMDP(ra.task, ra.Cfg)
	ra.qt = discrete.NewQTable(ra.task.Obs.Ints.Count(), ra.task.Act.Ints.Count())
	ra.Cfg.RFoo = ra.GetRFoo(ra.task)
	ra.rmdp.RFoo = ra.Cfg.RFoo
}
func (ra *BebAgent) AgentStart(obs rlglue.Observation) (act rlglue.Action) {
	ra.lastState = discrete.State(ra.task.Obs.Ints.Index(obs.Ints()))
	act = rlglue.NewAction(ra.task.Act.Ints.Values(ra.qt.Pi(ra.lastState).Hashcode()), []float64{}, []byte{})
	ra.lastAction = discrete.Action(ra.task.Act.Ints.Index(act.Ints()))
	return
}
func (ra *BebAgent) AgentStep(reward float64, obs rlglue.Observation) (act rlglue.Action) {
	nextState := discrete.State(ra.task.Obs.Ints.Index(obs.Ints()))
	learned := ra.rmdp.Observe(ra.lastState, ra.lastAction, nextState, reward)
	if learned {
		vi.ValueIteration(ra.qt, ra.rmdp, ra.Cfg.Epsilon)
	}
	ra.lastState = nextState
	act = rlglue.NewAction(ra.task.Act.Ints.Values(ra.qt.Pi(ra.lastState).Hashcode()), []float64{}, []byte{})
	ra.lastAction = discrete.Action(ra.task.Act.Ints.Index(act.Ints()))
	return
}
func (ra *BebAgent) AgentEnd(reward float64) {
	learned := ra.rmdp.ObserveTerminal(ra.lastState, ra.lastAction, reward)
	if learned {
		vi.ValueIteration(ra.qt, ra.rmdp, ra.Cfg.Epsilon)
	}
}
func (ra *BebAgent) AgentCleanup() {
}
func (ra *BebAgent) AgentMessage(message string) string {
	return ""
}
