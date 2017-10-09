package template;

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

public class ReactiveInstant implements ReactiveBehavior {

	private Random random;
	private int numActions;
	private Agent myAgent;
	
	
	private double[] cityExpectation;
	private City[] strategy;

	@Override
	public void setup(Topology topology, TaskDistribution td, Agent agent) {


		this.random = new Random();
		this.numActions = 0;
		this.myAgent = agent;
		
		int costPerKm = agent.vehicles().get(0).costPerKm();
		
		cityExpectation = new double[topology.size()];
		for(City city1:topology) {
			
			double expectation = 0;
			for(City city2:topology) {
				expectation+=td.probability(city1, city2)*td.reward(city1, city2);
			}
			cityExpectation[city1.id]=expectation;
		}
		
		strategy = new City[topology.size()];
		
		for(City city1: topology) {
			double best = Double.NEGATIVE_INFINITY;
			for(City city2: city1.neighbors()) {
				if(cityExpectation[city2.id]-city1.distanceTo(city2)*costPerKm>best) {
					best=cityExpectation[city2.id]-city1.distanceTo(city2)*costPerKm;
					strategy[city1.id]=city2;
				}
			}
		}
		
		
	}

	@Override
	public Action act(Vehicle vehicle, Task availableTask) {
		Action action;

		City currentCity = vehicle.getCurrentCity();
		
		
		
		if (availableTask == null || availableTask.reward<cityExpectation[strategy[currentCity.id].id]) {
			action = new Move(strategy[currentCity.id]);
		} else {
			action = new Pickup(availableTask);
		}
		
		if (numActions % 1000 == 0) {
			System.out.println("Instant ("+vehicle.name()+") ["+numActions+"] = "+myAgent.getTotalProfit()+" km:"+myAgent.getTotalDistance());
		}
		numActions++;
		
		return action;
	}
}
