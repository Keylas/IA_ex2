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

public class ReactiveRLA implements ReactiveBehavior {

	private Random random;
	private double discount;
	private int numActions;
	private Agent myAgent;


	/* NOTASK and PICKUP just help readability when it comes to action/task availability
	 * They will actually both be equals to the number of city (so must be read from configuration)
	 */
	private int NOTASK;
	private int PICKUP;

	/*Keep a reference to the cities for convenience
	 * Actual use is when creating the Move action
	 */
	private City[] cities;

	/*
	 * Strategy of the agent =  strategy(s)
	 * a state is {currentCity} x {deliveryCityForTask} (with destination possibly null so numCity*(numCity+1) possible states
	 * When it has to take a decision, the agent applies strategy(s)
	 */
	private int[][] strategy;


	@SuppressWarnings("unchecked")
	public void setup(Topology topology, TaskDistribution td, Agent agent) {


		long startTime = System.currentTimeMillis();
		long limiTime = agent.readProperty("timeout-setup", Long.class, 5000L);

		// Reads the discount factor from the agents.xml file.
		// If the property is not present it defaults to 0.95
		this.discount = agent.readProperty("discount-factor", Double.class,0.95);

		this.random = new Random();
		this.numActions = 0;
		this.myAgent = agent;

		//Get the costPerKm of our vehicle
		int costPerKm = agent.vehicles().get(0).costPerKm();



		int numCities = topology.size();
		NOTASK=numCities;
		PICKUP=numCities;



		//System.out.println(numCities);

		cities = new City[numCities];

		/*
		 * Extract and store information from the topology and task distribution
		 * We make a copy for readability
		 */

		double[][] costBetweenCities = new double[numCities][numCities];

		/*
		 * probabilityForTask[a][b] is the probability that there is a task for city b in city a, or no task if b=NOTASK=numCity
		 * (city are from 0 to numCity-1 so numCity is used for NOTASK)
		 */
		double[][] probabilityForTask = new double[numCities][numCities+1];

		double[][] taskReward = new double[numCities][numCities];




		//Fill information tables
		for(City city1:topology) {

			//keep a reference to the cities
			cities[city1.id]=city1;

			for(City city2:topology) {				
				costBetweenCities[city1.id][city2.id]=city1.distanceTo(city2)*costPerKm; //fill cost table between cities [numCity*numCity]
				probabilityForTask[city1.id][city2.id]=td.probability(city1, city2); //fill probability for task between city1 and city2
				taskReward[city1.id][city2.id]=td.reward(city1, city2); //fill the expected reward for the task

			}

			// also fill the probability for no task in city1
			probabilityForTask[city1.id][numCities]=td.probability(city1, null);
		}

		//
		//Print world properties
		System.out.println("Moving Costs");
		printV(costBetweenCities);
		System.out.println("Rewards");
		printV(taskReward);
		System.out.println("Probabilities");
		printV(probabilityForTask);
		 //

		//instantiate our V(S) that give the value of states
		double[][] valueOfState = new double[numCities][numCities+1];

		//Create Q(S,a): for each numCity*(numCity+1) state [][] , we have a list of possible actions that give a vaule (so it is Hasmap<actioncode> => Q(s,a))
		HashMap<Integer,Double>[][] q = (HashMap<Integer,Double>[][]) Array.newInstance(HashMap.class, topology.size(),topology.size()+1);

		//Instantiate Q(S) & V(S)

		//Double loop on the cities and the possible tasks <=> loop on the states
		for(City cCity: topology) {
			for(int deliveryCity=0; deliveryCity<numCities+1; deliveryCity++) {

				/*For every state, create the set of possible actions and their result
				 * Action = move to city_i which is in currentCity.neighbors() (action i)
				 * or pickup task (action PICKUP)
				 * [We use numCity as PICKUP code since cities goes from 0 to numCity
				 */
				HashMap<Integer,Double> h = new HashMap<Integer,Double>();
				for(City nCity:cCity.neighbors()) {
					h.put(nCity.id, -costBetweenCities[cCity.id][nCity.id]);
				}
				if(deliveryCity!=NOTASK) {h.put(PICKUP,0.0);}	//if we are in a state with an available task, action pickup possible
				q[cCity.id][deliveryCity]=h;
			}
		}




		//LEARNING PHASE


		/*
		 * Control variables to check for convergence:
		 * we stop after a loop reached: forall s, |V(s)_after - V(s)_before|<epsilon
		 */
		double epsilon=0.1;
		if(numCities>1) {epsilon = costBetweenCities[0][1]/100000;} //index epsilon on the cost to travel, a somehow fair indicator
		int numberOfStatesUnchanged =0; //when to stop: no value of S(V) has changed of more than epsilon 


		while(numberOfStatesUnchanged!=numCities*(numCities+1) && System.currentTimeMillis()-startTime<limiTime*0.8) {
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
							value+=taskReward[currentCity][deliveryCity];
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
						double sumOnTransition=0;
						for(int taskAtNext=0; taskAtNext<numCities+1; taskAtNext++) {
							sumOnTransition+=probabilityForTask[nextCity][taskAtNext]*valueOfState[nextCity][taskAtNext];
						}

						value+=discount*sumOnTransition;

						//Q(s,a)<-R(s,a) + gamma*sum{s}(T(s,a,s')V(s'))
						q[currentCity][deliveryCity].replace(action, value);

						//Keep track of max{a}(Q(s,a))
						if(value>max) {
							max=value;
						}
					}

					//check if we'll actually changed V(s)
					if(Math.abs(valueOfState[currentCity][deliveryCity]-max)<epsilon) {numberOfStatesUnchanged++;}
					//V(s)<-max{a}(Q(s,a))
					valueOfState[currentCity][deliveryCity]=max;
				}

			}
		}


		/*Fill the strategy:
		 * when in a state s=(city,task) [s]=[i][j]
		 * execute strategy(s)
		 */

		strategy = new int[numCities][numCities+1];


		//Loop for all states
		for(int currentCity=0; currentCity<numCities; currentCity++) {
			for(int deliveryCity=0; deliveryCity<numCities+1; deliveryCity++) {

				//Find what the best expected action is
				int bestAction=PICKUP;
				double bestResult=Double.NEGATIVE_INFINITY;
				for(int k:q[currentCity][deliveryCity].keySet()) {
					if(q[currentCity][deliveryCity].get(k)>bestResult) {
						bestResult = q[currentCity][deliveryCity].get(k);
						bestAction=k;
					}


					strategy[currentCity][deliveryCity]=bestAction;

				}
			}
		}

		//Print computed strategy
		System.out.println("Strategy "+discount);
		printV(strategy);


	}

	@Override
	public Action act(Vehicle vehicle, Task availableTask) {
		Action action;

		City currentCity = vehicle.getCurrentCity();

		if(availableTask==null || strategy[currentCity.id][availableTask.deliveryCity.id]!=PICKUP) {

			action = new Move(cities[strategy[currentCity.id][NOTASK]]);

		} else {
			action = new Pickup(availableTask);
		}



		if (numActions % 10000 == 0) {
			System.out.println("RLA {"+discount+"} ("+vehicle.name()+") ["+numActions+"] = "+myAgent.getTotalProfit()+" km:"+myAgent.getTotalDistance());
		}
		numActions++;

		return action;
	}


	//Utility to print debug
	public void printV(double[][] v) {
		String s= "";
		for(int i=0; i<v.length; i++) {
			for(int j=0; j<v[i].length; j++) {
				s+=(v[i][j])+" ";
			}
			s+='\n';
		}
		System.out.println(s);
	}

	public void printV(int[][] v) {
		String s= "";
		for(int i=0; i<v.length; i++) {
			for(int j=0; j<v[i].length; j++) {
				s+=(v[i][j])+" ";
			}
			s+='\n';
		}
		System.out.println(s);
	}

	public void printV(int[] v) {
		String s="";
		for(int i=0; i<v.length; i++) {
			s+=(v[i])+" ";
		}
		System.out.println(s);
	}

}
