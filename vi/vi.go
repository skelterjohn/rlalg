/* 
* Copyright (C) 2010, John Asmuth

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
* 
*  $Revision$
*  $Date$
*  $Author$
*  $HeadURL$
* 
*/


package vi

import (
	"math"
	"go-glue.googlecode.com/hg/rltools/discrete"
	//"fmt"
	//"os"
)

func BackupStateAction(qt *discrete.QTable, mdp discrete.MDP, s discrete.State, a discrete.Action) (error float64) {
	var nq float64

	for n := range mdp.S64() {
		ev := mdp.T(s, a, n)
		ev *= qt.V(n)
		nq += ev
	}
	nq *= mdp.GetGamma()
	nq += mdp.R(s, a)

	error = math.Fabs(nq - qt.Q(s, a))

	qt.SetQ(s, a, nq)

	return
}

func ValueIteration(qt *discrete.QTable, mdp discrete.MDP, epsilon float64) (numIterations int) {
	//fmt.Fprintf(os.Stderr, "+ValueIteration\n")
	//fmt.Println(mdp.GetGamma())
	//defer fmt.Fprintf(os.Stderr, "-ValueIteration\n")
	var error float64
	for {
		numIterations += 1
		//fmt.Printf("iteration %d\n", numIterations)
		error = 0
		for s := range mdp.S64() {
			for a := range mdp.A64() {
				saError := BackupStateAction(qt, mdp, s, a)
				error = math.Fmax(error, saError)
			}
		}
		//fmt.Printf("QT\n%v\n", qt)
		//fmt.Fprintf(os.Stderr, "error %f\n%v\n", error, qt)
		if error < epsilon {
			return
		}
	}
	return
}
