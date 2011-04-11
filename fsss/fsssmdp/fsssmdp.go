package fsssmdp

import (
	"strings"
	"strconv"
	"rand"
	"gostat.googlecode.com/hg/stat"
	"go-glue.googlecode.com/hg/rlglue"
	"go-glue.googlecode.com/hg/rltools/discrete"
	"github.com/skelterjohn/rlalg/fsss"
)

type Config struct {
	Depth		uint64
	NumTrajectories	uint64
	Budget		uint64
	FS3		fsss.Config
}

func ConfigDefault() (cfg Config) {
	cfg.Depth = 10
	cfg.NumTrajectories = 100
	cfg.Budget = 1000
	cfg.FS3 = fsss.ConfigDefault()
	return
}

type Agent struct {
	cfg			Config
	mdp			discrete.MDP
	lastState		discrete.State
	lastAction		discrete.Action
	s			*fsss.Searcher
	mdpo			*discrete.MDPOracle
	stepsWithPlanner	uint64
}

func New(cfg Config, mdp discrete.MDP) (this *Agent) {
	this = new(Agent)
	this.cfg = cfg
	this.mdp = mdp
	this.mdpo = discrete.NewMDPOracle(this.mdp, 0)
	this.s = fsss.New()
	this.s.Cfg = this.cfg.FS3
	this.s.NumActions = this.mdp.NumActions()
	this.s.Gamma = mdp.GetGamma()
	this.s.Vmin = this.mdp.GetTask().Reward.Min / (1 - this.s.Gamma)
	this.s.Vmax = this.mdp.GetTask().Reward.Max / (1 - this.s.Gamma)
	this.stepsWithPlanner = 0
	return
}
func (*Agent) AgentInit(taskString string) {
}
func (this *Agent) AgentStart(obs rlglue.Observation) (act rlglue.Action) {
	this.stepsWithPlanner = 0
	this.lastState = discrete.State(this.mdp.GetTask().Obs.Ints.Index(obs.Ints()))
	this.Plan()
	act = rlglue.NewAction(this.mdp.GetTask().Act.Ints.Values(this.GetAction()), []float64{}, []byte{})
	this.lastAction = discrete.Action(this.mdp.GetTask().Act.Ints.Index(act.Ints()))
	return
}
func (this *Agent) AgentStep(reward float64, obs rlglue.Observation) (act rlglue.Action) {
	this.stepsWithPlanner++
	nextState := discrete.State(this.mdp.GetTask().Obs.Ints.Index(obs.Ints()))
	this.lastState = nextState
	this.Plan()
	act = rlglue.NewAction(this.mdp.GetTask().Act.Ints.Values(this.GetAction()), []float64{}, []byte{})
	this.lastAction = discrete.Action(this.mdp.GetTask().Act.Ints.Index(act.Ints()))
	return
}
func (this *Agent) AgentEnd(reward float64) {
}
func (this *Agent) AgentCleanup() {
}
func (this *Agent) AgentMessage(message string) string {
	tokens := strings.Split(message, " ", -1)
	if tokens[0] == "seed" {
		seed, _ := strconv.Atoi64(tokens[1])
		rand.Seed(seed)
	}
	return ""
}
func (this *Agent) GetAction() (action uint64) {
	if this.s == nil {
		action = uint64(stat.NextRange(int64(this.mdp.GetTask().Act.Ints.Count())))
		return
	}
	node := this.s.GetNode(this.stepsWithPlanner, this.mdpo)
	action = uint64(this.s.GetAction(node))
	this.s.ClearLevel(this.stepsWithPlanner)
	return
}
func (this *Agent) Plan() {
	this.mdpo = this.mdpo.Teleport(this.lastState)
	root := this.s.GetNode(this.stepsWithPlanner, this.mdpo)
	var expanded uint64
	for i := 0; i < int(this.cfg.NumTrajectories); i++ {
		expanded += this.s.RunTrajectory(root, this.cfg.Depth)
		if this.cfg.Budget != 0 && expanded > this.cfg.Budget {
			break
		}
	}
}
