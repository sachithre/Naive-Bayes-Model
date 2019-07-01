import java.io.BufferedReader;
import java.io.FileReader;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class LensePredict {

	public static void main(String[] args) throws Exception {
		FileReader reader = new FileReader("C:\\Users\\Sachithre\\eclipse-workspace\\lense.arff"); // Location of the arff File
		Instances instance_train = new Instances(reader); //create an instance
		BufferedReader fi = new BufferedReader(reader); //read the file and pass data to buffer
		fi.close(); // file close
		
		instance_train.setClassIndex(4); //we are gonna set 5th attribute as class
		NaiveBayes n = new NaiveBayes();
		
		n.buildClassifier(instance_train); //load data to weka built in function
		System.out.println("Naive Bayes model");
		System.out.println(n.toString());
		
		Instance inst = new DenseInstance(4);
		inst.setDataset(instance_train);
		
		inst.setValue(0, "pre-presbyopic"); 
		inst.setValue(1, "hypermetrope"); 
		inst.setValue(2, "no"); 
		inst.setValue(3, "normal"); 
		
		System.out.println("The instance: " + inst); 
		System.out.println(n.classifyInstance(inst));

	}

}
