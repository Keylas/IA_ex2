package template;

import java.lang.reflect.Array;
import java.util.HashMap;
import java.util.Random;
import logist.simulation.Vehicle;
import logist.agent.Agent;
import logist.behavior.ReactiveBehavior;
import logist.plan.Action;
import logist.plan.Action.Move;
import logist.plan.Action.Pickup;
import logist.task.Task;
import logist.task.TaskDistribution;
import logist.topology.Topology;
import logist.topology.Topology.City;

public class ReactiveUncheck implements ReactiveBehavior {

	private Random random;
	private double pPickup;
	private int numActions;
	private Agent myAgent;


	
	private int NOTASK;
	private int PICKUP;
	
	private City[] cities;

	private int[][] strategy;


	@SuppressWarnings("unchecked")
	public void setup(Topology topology, TaskDistribution td, Agent agent) {



		// Reads the discount factor from the agents.xml file.
		// If the property is not present it defaults to 0.95
		Double discount = agent.readProperty("discount-factor", Double.class,
				0.95);

		int costPerKm = agent.vehicles().get(0).costPerKm();

		double epsilon = 0.1;


		int numCities = topology.size();
		NOTASK=numCities;
		PICKUP=numCities;

		System.out.println(numCities);

		cities = new City[numCities];
		
		double[][] costBetweenCities = new double[numCities][numCities];

		double[][] probabilityForTask = new double[numCities][numCities+1];

		double[][] expectedTaskReward = new double[numCities][numCities];


		

		//Fill information tables
		for(City city1:topology) {
			
			cities[city1.id]=city1;
			
			for(City city2:topology) {
				costBetweenCities[city1.id][city2.id]=city1.distanceTo(city2)*costPerKm;
				probabilityForTask[city1.id][city2.id]=td.probability(city1, city2);
				expectedTaskReward[city1.id][city2.id]=td.reward(city1, city2);

			}
			probabilityForTask[city1.id][numCities]=td.probability(city1, null);
		}


		//V(S)
		double[][] valueOfState = new double[numCities][numCities+1];

		//Q(S)
		HashMap<Integer,Double>[][] q = (HashMap<Integer,Double>[][]) Array.newInstance(HashMap.class, topology.size(),topology.size()+1);

		//Instantiate Q(S) & V(S)

		//Double loop on the cities and the possible tasks <=> loop on the states
		for(City cCity: topology) {
			for(int deliveryCity=0; deliveryCity<numCities+1; deliveryCity++) {

				/*For every state, create the set of possible actions and their result
				 * Action = move to city_i which is in currentCity.neighbors() (action i)
				 * or pickup task (action PICKUP)
				 */
				HashMap<Integer,Double> h = new HashMap<Integer,Double>();
				for(City nCity:cCity.neighbors()) {
					h.put(nCity.id, -costBetweenCities[cCity.id][nCity.id]);
				}
				if(deliveryCity!=NOTASK) {h.put(PICKUP,0.0);}	//we are in a state with an available task: action pickup possible
				q[cCity.id][deliveryCity]=h;
			}
		}


		//LEARNING 


		int numberOfStatesUnchanged =0; //when to stop: no value of S(V) has change of more than epsilon 
		while(numberOfStatesUnchanged!=numCities*(numCities+1)) {
			numberOfStatesUnchanged=0;

			//Loop on the possible states (s in S)
			for(int currentCity=0; currentCity<numCities; currentCity++) {
				for(int deliveryCity=0; deliveryCity<numCities+1; deliveryCity++) {



					double max = Double.NEGATIVE_INFINITY;

					//Loop on action possible (on state S) a in A
					for(int action:q[currentCity][deliveryCity].keySet()) {

						double value=0;


						int nextCity;

						if(action==PICKUP) {
							//if the action is pickup, we'll be moved to the deliveryCity and get a reward
							value+=expectedTaskReward[currentCity][deliveryCity];
							nextCity=deliveryCity;
						} else {
							//else our action is to go to a new city
							nextCity=action;
						}
						//either way we pay to move
						value-=costBetweenCities[currentCity][nextCity];

						/*At this point value=R(s,a)=reward(s,a)-cost(s,a)
						 * where reward is actually 0 if we didn't pickup
						 */


						//we now compute gamma*sum{s}(transitionProbability*V(S))
						double maybe=0;
						for(int taskAtNext=0; taskAtNext<numCities+1; taskAtNext++) {
							maybe+=probabilityForTask[nextCity][taskAtNext]*valueOfState[nextCity][taskAtNext];
						}

						value+=discount*maybe;

						//Q(s,a)<-R(s,a) + gamma*sum{s}(T(s,a,s')V(s'))
						q[currentCity][deliveryCity].replace(action, value);

						//Keep track of max{a}(Q(s,a))
						if(value>max) {
							max=value;
						}
					}

					//V(S)<-max{a}(Q(s,a))
					//also check if we actually changed V(S)
					if(Math.abs(valueOfState[currentCity][deliveryCity]-max)<epsilon) {numberOfStatesUnchanged++;}
					valueOfState[currentCity][deliveryCity]=max;
				}

			}			
		}


		/*Fill the strategy:
		 * when in a state s=(city,task) [s]=[i][j]
		 * either pickup task if the reward is above threshold[s]
		 * or move to the best city with is strategy[s]
		 */

		strategy = new int[numCities][numCities+1];

		//Loop for all states
		for(int currentCity=0; currentCity<numCities; currentCity++) {
			for(int depositCity=0; depositCity<numCities+1; depositCity++) {

				//Find again what the best EXPECTED action is : max{a}Q(s,a)
				int bestAction=PICKUP;
				double bestResult=Double.NEGATIVE_INFINITY;
				for(int k:q[currentCity][depositCity].keySet()) {
					if(q[currentCity][depositCity].get(k)>bestResult) {
						bestResult = q[currentCity][depositCity].get(k);
						bestAction=k;
					}
					
					strategy[currentCity][depositCity]=bestAction;
				}
			}
		}
		



		this.random = new Random();
		this.pPickup = discount;
		this.numActions = 0;
		this.myAgent = agent;

	}

	@Override
	public Action act(Vehicle vehicle, Task availableTask) {
		Action action;

		City currentCity = vehicle.getCurrentCity();
		
		//action = read_learnt(vehicule.position, task ) 

		if(availableTask==null) {
			
			action = new Move(cities[strategy[currentCity.id][NOTASK]]);
			

		} else {
			action = new Pickup(availableTask);
		}


		//TODO
		if (numActions % 1000 == 0) {
			System.out.println("Uncheck profit after "+numActions+" actions is "+myAgent.getTotalProfit());
		}
		numActions++;

		return action;
	}


}
