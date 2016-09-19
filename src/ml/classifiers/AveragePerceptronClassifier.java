package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;

import ml.DataSet;
import ml.Example;

public class AveragePerceptronClassifier extends PerceptronClassifier {
	private double b2 = 0;
	private double[] aggweights;
	
	public AveragePerceptronClassifier() {
		
	}
	
	@Override
	public void train(DataSet data) {
		Set<Integer> myFeatureSet = data.getAllFeatureIndices();
		weights = new double[myFeatureSet.size()];
		aggweights = new double[myFeatureSet.size()];
		int updated = 0;
		int total = 0;
		ArrayList<Example> myExamples = data.getData();
		for (int i=0; i<iterations; i++) {
			Collections.shuffle(myExamples);
			for (Example e : myExamples) {
				double prediction = b + sumWeightedFeatures(e);
				if (prediction * e.getLabel() <= 0) {
					updateWeightsAndB(e);
					updateAggWeightsAndB2(updated);
					updated = 0;
				}
				updated++;
				total++;
			}
		}
		updateAggWeightsAndB2(updated);
		int myIndexCounter = 0;
		for (double d : aggweights) {
			aggweights[myIndexCounter] = aggweights[myIndexCounter]/total;
			myIndexCounter++;
		}
		b2 = b2/total;
		weights = aggweights;
		b = b2;
	}
	
	/**
	 * Updates aggregate weights and b-value on misclassification.
	 * 
	 * @param updated	Number of consecutive successful classifications before failure.
	 */
	private void updateAggWeightsAndB2(int updated) {
		int myIndexCounter = 0;
		for (double d : aggweights) {
			aggweights[myIndexCounter] += (weights[myIndexCounter]*updated);
			myIndexCounter++;
		}
		b2 += b*updated;
	}
	
	/**
	 * "Resets" the classifier, for testing purposes.
	 */
	@Override
	protected void forgetTraining() {
		weights = new double[weights.length];
		aggweights = new double[aggweights.length];
		b = 0;
		b2 = 0;
	}
	
	/**
	 * Main included primarily for sout accuracy testing.
	 * 
	 * @param args	N/A
	 */
	public static void main(String[] args) {
		String csvFile = "C:/Users/Nick/Documents/School/Pomona College/Sr 1st Sem/Eclipse Workspace/cs158_assignment3/src/ml/titanic-train.perc.csv";
		DataSet dataset = new DataSet(csvFile);
		AveragePerceptronClassifier original = new AveragePerceptronClassifier();
		double myAccuracy = testClassifier(original, dataset);
		System.out.println("Accuracy was: " + myAccuracy);
	}
}
