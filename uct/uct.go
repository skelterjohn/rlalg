package uct

import (
	"math"
	"gohash.googlecode.com/hg/hashlessmap"
)

type Config struct {
	Beta  float64
	Gamma float64
}

func ConfigDefault() (cfg Config) {
	cfg.Beta = 1
	return
}

type Oracle interface {
	//needs to be look-upable
	hashlessmap.HasherLess
	Next(action uint64) (o Oracle, r float64)
	Terminal() bool
	//Str() string
}

type ActionFilter interface {
	ActionAvailable(action uint64) bool
}

type Node struct {
	cfg *Config

	R           []float64
	Q           []float64
	QBonus      []float64
	V           float64
	Branches    []map[*Node]float64
	Visits      []int
	TotalVisits int
}

func GetNode(o Oracle) (this *Node) {
	return
}

func UCBBonus(visits, totalVisits int) (bonus float64) {
	bonus = math.Sqrt(math.Log(float64(visits)) / float64(totalVisits))
	return
}

func (this *Node) Backup() {
	this.V = math.Log(0)

	for a := range this.Q {
		for child, count := range this.Branches[a] {
			this.Q[a] += child.V * count
		}
		this.Q[a] /= float64(this.Visits[a])
		this.Q[a] *= this.cfg.Gamma
		this.Q[a] += this.R[a]
		if this.Q[a] > this.V {
			this.V = this.Q[a]
		}
		this.QBonus[a] = this.cfg.Beta * UCBBonus(this.Visits[a], this.TotalVisits)
	}
}
