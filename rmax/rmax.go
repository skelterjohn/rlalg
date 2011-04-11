package rmax

import (
	"go-glue.googlecode.com/hg/rlglue"
	"go-glue.googlecode.com/hg/rltools/discrete"
	"github.com/skelterjohn/rlalg/vi"
)

type RmaxMDP struct {
	discrete.FlatMDP
	CountsSA	[][]int
	CountsSAN	[][][]int
	TotalR		[][]float64
	Vmax		float64
	M		int
}

func NewRmaxMDP(task *rlglue.TaskSpec, m uint64) (rm *RmaxMDP) {
	numStates := task.Obs.Ints.Count()
	numActions := task.Act.Ints.Count()
	rm = new(RmaxMDP)
	rm.Task = task
	rm.Transitions = make([][][]float64, numStates)
	rm.Rewards = make([][]float64, numStates)
	rm.Gamma = task.DiscountFactor
	rm.CountsSA = make([][]int, numStates)
	rm.CountsSAN = make([][][]int, numStates)
	rm.TotalR = make([][]float64, numStates)
	for s := range rm.CountsSA {
		rm.CountsSA[s] = make([]int, numActions)
		rm.CountsSAN[s] = make([][]int, numActions)
		for a, _ := range rm.CountsSAN[s] {
			rm.CountsSAN[s][a] = make([]int, numStates)
		}
		rm.TotalR[s] = make([]float64, numActions)
		rm.Transitions[s] = make([][]float64, numActions)
		for a, _ := range rm.Transitions[s] {
			rm.Transitions[s][a] = make([]float64, numStates)
		}
	}
	if rm.Gamma < 1 {
		rm.Vmax = task.Reward.Max / (1 - rm.Gamma)
	} else {
		rm.Vmax = task.Reward.Max
	}
	rm.M = int(m)
	return
}
func (rm *RmaxMDP) T(s discrete.State, a discrete.Action, n discrete.State) float64 {
	if rm.CountsSA[s][a] < rm.M {
		return 0
	}
	return rm.Transitions[s][a][n]
}
func (rm *RmaxMDP) R(s discrete.State, a discrete.Action) float64 {
	if rm.CountsSA[s][a] < rm.M {
		return rm.Vmax
	}
	return rm.Rewards[s][a]
}
func (rm *RmaxMDP) resolve(s discrete.State, a discrete.Action) {
	if rm.Transitions[s] == nil {
		rm.Transitions[s] = make([][]float64, len(rm.CountsSAN[s]))
	}
	if rm.Transitions[s][a] == nil {
		rm.Transitions[s][a] = make([]float64, len(rm.CountsSAN[s][a]))
	}
	if rm.Rewards[s] == nil {
		rm.Rewards[s] = make([]float64, len(rm.TotalR[s]))
	}
	norm := 1.0 / float64(rm.CountsSA[s][a])
	for n, count := range rm.CountsSAN[s][a] {
		p := float64(count) * norm
		rm.Transitions[s][a][n] = p
	}
	rm.Rewards[s][a] = rm.TotalR[s][a] * norm
}
func (rm *RmaxMDP) Observe(s discrete.State, a discrete.Action, n discrete.State, r float64) (learned bool) {
	rm.CountsSA[s][a]++
	rm.CountsSAN[s][a][n]++
	rm.TotalR[s][a] += r
	learned = rm.CountsSA[s][a] == rm.M
	if learned {
		rm.resolve(s, a)
	}
	return
}
func (rm *RmaxMDP) ObserveTerminal(s discrete.State, a discrete.Action, r float64) (learned bool) {
	rm.CountsSA[s][a]++
	rm.TotalR[s][a] += r
	learned = rm.CountsSA[s][a] == rm.M
	if learned {
		rm.resolve(s, a)
	}
	return
}

type RmaxConfig struct {
	M	uint64
	Epsilon	float64
}

func RmaxConfigDefault() (cfg RmaxConfig) {
	cfg.M = 5
	cfg.Epsilon = 0.1
	return
}

type RmaxAgent struct {
	task		*rlglue.TaskSpec
	rmdp		*RmaxMDP
	qt		*discrete.QTable
	lastState	discrete.State
	lastAction	discrete.Action
	Cfg		RmaxConfig
}

func NewRmaxAgent(Cfg RmaxConfig) (ra *RmaxAgent) {
	ra = new(RmaxAgent)
	ra.Cfg = Cfg
	return
}
func (ra *RmaxAgent) AgentInit(taskString string) {
	ra.task, _ = rlglue.ParseTaskSpec(taskString)
	ra.rmdp = NewRmaxMDP(ra.task, ra.Cfg.M)
	ra.qt = discrete.NewQTable(ra.task.Obs.Ints.Count(), ra.task.Act.Ints.Count())
}
func (ra *RmaxAgent) AgentStart(obs rlglue.Observation) (act rlglue.Action) {
	ra.lastState = discrete.State(ra.task.Obs.Ints.Index(obs.Ints()))
	act = rlglue.NewAction(ra.task.Act.Ints.Values(ra.qt.Pi(ra.lastState).Hashcode()), []float64{}, []byte{})
	ra.lastAction = discrete.Action(ra.task.Act.Ints.Index(act.Ints()))
	return
}
func (ra *RmaxAgent) AgentStep(reward float64, obs rlglue.Observation) (act rlglue.Action) {
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
func (ra *RmaxAgent) AgentEnd(reward float64) {
	learned := ra.rmdp.ObserveTerminal(ra.lastState, ra.lastAction, reward)
	if learned {
		vi.ValueIteration(ra.qt, ra.rmdp, ra.Cfg.Epsilon)
	}
}
func (ra *RmaxAgent) AgentCleanup() {
}
func (ra *RmaxAgent) AgentMessage(message string) string {
	return ""
}
