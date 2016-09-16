// Nick Reminder, Maddie Gordon
// cs158 Assignment 3

package ml.classifiers;

import static java.util.Collections.shuffle;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;

import ml.DataSet;
import ml.Example;

/**
 * Implements a basic perceptron classifier.
 * 
 * @author Nick Reminder, Maddie Gordon
 *
 */
public class PerceptronClassifier implements Classifier {
	private int iterations = 10;
	private double b = 0;
	private double[] weights;
	
	public PerceptronClassifier() {
	}
	
	/**
	 * Trains classifier model on DataSet argument.
	 */
	@Override
	public void train(DataSet data) {
		Set<Integer> myFeatureSet = data.getAllFeatureIndices();
		weights = new double[myFeatureSet.size()];
		ArrayList<Example> myExamples = data.getData();
		shuffle(myExamples);
		for (int i=0; i<iterations; i++) {
			for (Example e : myExamples) {
				double prediction = b + sumWeightedFeatures(e);
				if (prediction * e.getLabel() <= 0) {
					updateWeightsAndB(e);
				}
			}
		}
	}

	/**
	 * Classifies an example based on trained model. 
	 * 
	 * @return	Positive numbers are positive, negative numbers are negative.
	 */
	@Override
	public double classify(Example example) {
		return sumWeightedFeatures(example) + b;
	}
	
	/**
	 * Specifies number of iterations over Example set in training.
	 * 
	 * @param aIterations	Sets the number of iterations (default 10).
	 */
	public void setIterations(int aIterations) {
		iterations = aIterations;
	}
	
	/**
	 * Returns String representation of PerceptronClassifier.
	 * 
	 * @return	Said String representation.
	 */
	public String toString() {
		String theReturn = new String();
		for (int i=0; i<weights.length; i++) {
			theReturn.concat(i + ":" + weights[i] + " ");
		}
		return theReturn + b;
	}

	/**
	 * Summation of feature * corresponding weight for all features in a given product.
	 * Positive values are positive, negative values are negative.
	 * 
	 * @param aExample	The example in question.
	 * @return			Double representing prediction for the model.
	 */
	private double sumWeightedFeatures(Example aExample) {
		Set<Integer> myFeatureSet = aExample.getFeatureSet();
		double myCarrier = 0;
		int myIndexCounter = 0;
		for (int i : myFeatureSet) {
			myCarrier += aExample.getFeature(i)*weights[myIndexCounter];
			myIndexCounter++;
		}
		return myCarrier;
	}
	
	/**
	 * Updates weight vector and intercept given example-model disagreement.
	 *
	 * @param aExample	The example that broke our model.
	 */
	private void updateWeightsAndB(Example aExample) {
		Set<Integer> myFeatureSet = aExample.getFeatureSet();
		int myIndexCounter = 0;
		for (int i : myFeatureSet) {
			weights[myIndexCounter] += (aExample.getFeature(i)*aExample.getLabel());
			myIndexCounter++;
		}
		b += aExample.getLabel();
	}
	
	/**
	 * Main included primarily for sout accuracy testing.
	 * 
	 * @param args	N/A
	 */
	public static void main(String[] args) {
		String csvFile = "C:/Users/Nick/Documents/School/Pomona College/Sr 1st Sem/Eclipse Workspace/cs158_assignment3/src/ml/titanic-train.perc.csv";
		DataSet dataset = new DataSet(csvFile);
		PerceptronClassifier original = new PerceptronClassifier();
		double myAccuracy = testClassifier(original, dataset);
		System.out.println("Accuracy was: " + myAccuracy);
	}
	
	/**
	 * Tests a trained classifier over a testing dataset, returning decimal accuracy.
	 * 
	 * @param aClassifier	Classifier to be tested.
	 * @param aDataSet		Test dataset.
	 * @return				Decimal representation of accuracy, [0,1].
	 */
	public static double testClassifier(Classifier aClassifier, DataSet aDataSet) {
		double cumulativeAccuracy = 0;
		for (int i=0; i<100; i++) {
			DataSet[] temp = aDataSet.split(0.8);
			aClassifier.train(temp[0]);
			ArrayList<Example> myExamples = temp[1].getData();
			double myEvaluated = 0;
			double myCorrect = 0;
			for(Example e : myExamples) {
				if (aClassifier.classify(e)*e.getLabel() > 0) { myCorrect++; } myEvaluated++;
			}
			cumulativeAccuracy += myCorrect/myEvaluated;
		}
		return cumulativeAccuracy/100;
	}
}
