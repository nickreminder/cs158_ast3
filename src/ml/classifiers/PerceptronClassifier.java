// Nick Reminder, Maddie Gordon
// cs158 Assignment 3

package ml.classifiers;

import static java.util.Collections.shuffle;
import static ml.classifiers.ClassifierTimer.timeClassifier;

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
	protected int iterations = 10;
	protected double b = 0;
	protected double[] weights;
	
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
		for (int i=0; i<iterations; i++) {
			shuffle(myExamples);
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
	protected double sumWeightedFeatures(Example aExample) {
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
	protected void updateWeightsAndB(Example aExample) {
		Set<Integer> myFeatureSet = aExample.getFeatureSet();
		int myIndexCounter = 0;
		for (int i : myFeatureSet) {
			weights[myIndexCounter] += (aExample.getFeature(i)*aExample.getLabel());
			myIndexCounter++;
		}
		b += aExample.getLabel();
	}
	
	/**
	 * "Resets" the classifier, for testing purposes.
	 */
	protected void forgetTraining() {
		weights = new double[weights.length];
		b = 0;
	}
	
	/**
	 * Main included primarily for sout accuracy testing.
	 * 
	 * @param args	N/A
	 */
	 public static void main(String[] args) {
//		String csvFile = "C:/Users/Nick/Documents/School/Pomona College/Sr 1st Sem/Eclipse Workspace/cs158_assignment3/src/ml/titanic-train.perc.csv";
//		DataSet dataset = new DataSet(csvFile);
//		PerceptronClassifier original = new PerceptronClassifier();
//		double myAccuracy = testClassifier(original, dataset);
//		System.out.println("Accuracy was: " + myAccuracy);
//		original.forgetTraining();
//		timeClassifier(original, dataset, 1000);
		//testIterationOptimum(original, dataset);
	}
	
	/**
	 * Tests a trained classifier over a testing dataset, returning decimal accuracy.
	 * 
	 * @param aClassifier	Classifier to be tested.
	 * @param aDataSet		Test dataset.
	 * @return				Decimal representation of accuracy, [0,1].
	 */
	public static double testClassifier(PerceptronClassifier aClassifier, DataSet aDataSet) {
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
			aClassifier.forgetTraining();
		}
		return cumulativeAccuracy/100;
	}
	
	/**
	 * Does same as testClassifier, except also on training data, returning a string with both calculated accuracies.
	 * 
	 * @param aClassifier	Classifier to be evaluated.
	 * @param aDataSet		Data set to be evaluated.
	 * @return				String containing both accuracies.
	 */
	public static String testClassifierOnTestData(PerceptronClassifier aClassifier, DataSet aDataSet) {
		double cumulativeAccuracy = 0;
		double cumulativeAccuracy2 = 0;
		for (int i=0; i<100; i++) {
			DataSet[] temp = aDataSet.split(0.8);
			aClassifier.train(temp[0]);
			
			//Test on test data
			ArrayList<Example> myExamples = temp[1].getData();
			double myEvaluated = 0;
			double myCorrect = 0;
			for(Example e : myExamples) {
				if (aClassifier.classify(e)*e.getLabel() > 0) { myCorrect++; } myEvaluated++;
			}
			cumulativeAccuracy += myCorrect/myEvaluated;
			
			//Test on training data
			ArrayList<Example> myExamples2 = temp[0].getData();
			double myEvaluated2 = 0;
			double myCorrect2 = 0;
			for(Example e : myExamples2) {
				if (aClassifier.classify(e)*e.getLabel() > 0) { myCorrect2++; } myEvaluated2++;
			}
			cumulativeAccuracy2 += myCorrect2/myEvaluated2;
			
			aClassifier.forgetTraining();
		}
		return new String("Test data: " + (cumulativeAccuracy/100) + ", Training data: " + (cumulativeAccuracy2/100));
	}
	
	/**
	 * Uses sout to test effect of iteration number on accuracy.
	 * 
	 * @param aClassifier	Classifier to test.
	 * @param data			Data set to use for testing.
	 */
	public static void testIterationOptimum(PerceptronClassifier aClassifier, DataSet data) {
		aClassifier.forgetTraining();
		for (int i=1; i<50; i++) {
			aClassifier.setIterations(i);
			System.out.println("Accuracy for " + i + " iterations: " + testClassifierOnTestData(aClassifier, data));
		}
	}
}
