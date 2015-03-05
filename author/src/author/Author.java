/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package author;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.classifiers.functions.supportVector.RBFKernel;


/**
 *
 * @author Vishal
 */
public class Author {

    /**String arff="C:/Users/Vishal/Documents/wekadatasets/spambase(1).arff";
        FileReader fileReader = new FileReader(arff);
        Instances data = new Instances(fileReader);
        //SETTING INDEX
        data.setClassIndex(data.numAttributes()-1);
        //PRINTING CLASSSES ATTRIBUTES AND INSTANCES
        System.out.println("number of classes  " + data.numClasses());
        System.out.println("number of attributes  " +data.numAttributes());
        System.out.println("number of instances  " + data.numInstances());
        
        //NAIVEBAYES CLASSIFIER  
        Classifier naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(data);
        
        Evaluation eval = new Evaluation(data);
        // TEN FOLD AND TEN RANDOM
        int numFolds=10;
        eval.crossValidateModel(naiveBayes,data,numFolds,new Random(10));
        //PRINTING OUTPUTS FMEASURE,RECALL,PRECISION
        System.out.println("F measure is of spam is "+(eval.fMeasure(1)));
        System.out.println("Recall is  of spam is"+(eval.recall(1)));
        System.out.println("Precision is  of spam is"+(eval.precision(1)));
        System.out.println(eval.toSummaryString("\nresults",true));
        //J48 CLASSIFIER
        J48 tree = new J48();
        tree.buildClassifier(data);
        double classValue = tree.classifyInstance(data.instance(1)); 
        //data.instance(1).setClassValue(classValue);
        // CLASSVALUE OF FIRST INSTANCE
        System.out.println("   class value  is   "+ classValue);
        
        Evaluation evalj48 = new Evaluation(data);
        // NUMFOLDS 5 AND RANDOM 20
        int numFoldsj48=5;
        evalj48.crossValidateModel(tree,data,numFoldsj48,new Random(20));
        //FMEASURE RECALL AND PRECISION
        System.out.println("F measure is of spam in J48 is "+(evalj48.fMeasure(0)));
        System.out.println("Recall is  of spam in J48 is "+(evalj48.recall(0)));
        System.out.println("Precision is  of spam in J48 is"+(evalj48.precision(0)));
        System.out.println(evalj48.toSummaryString("\nresults in J48",true));
        
     * @param args the command line arguments
     */
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception  {
        // TODO code application logic here
        String arff="C:/Users/Vishal/Documents/wekadatasets/62programmers(1).arff";
        FileReader fileReader = new FileReader(arff);
        Instances data = new Instances(fileReader);
         
        //SETTING INDEX
        data.setClassIndex(data.numAttributes()-1);
        //PRINTING CLASSSES ATTRIBUTES AND INSTANCES
        System.out.println("number of classes  " + data.numClasses());
        System.out.println("number of attributes  " +data.numAttributes());
        System.out.println("number of instances  " + data.numInstances());
        
         
        // create new instance of scheme
 weka.classifiers.functions.SMO scheme = new weka.classifiers.functions.SMO();
 // set options
 weka.classifiers.functions.supportVector.RBFKernel RBF;
 RBF= new weka.classifiers.functions.supportVector.RBFKernel(data,250007,0.01);
 scheme.setKernel(RBF);
 //scheme.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\"")); 
 
 scheme.buildClassifier(data);
        
        
        Evaluation eval = new Evaluation(data);
        // TEN FOLD AND TEN RANDOM
        int numFolds=9;
        eval.crossValidateModel(scheme,data,numFolds,new Random(1));
        //PRINTING OUTPUTS FMEASURE,RECALL,PRECISION
        System.out.println("F measure is of spam is "+(eval.fMeasure(1)));
        System.out.println("Recall is  of spam is"+(eval.recall(1)));
        System.out.println("Precision is  of spam is"+(eval.precision(1)));
        System.out.println(eval.toSummaryString("\nresults",true));
       int numtree=100;
        int numfea=data.numAttributes();
        weka.classifiers.trees.RandomForest tree = new weka.classifiers.trees.RandomForest();
        //tree.setOptions(weka.core.Utils.splitOptions("weka.classifiers.trees.RandomForest -I 100 -K 0 -S 1"));
        tree.setNumTrees(numtree);
        //tree.setNumFeatures(numfea);
        tree.buildClassifier(data);
        
        
        System.out.println("features are and trees are "+numfea + ","+ numtree);
        //data.instance(1).setClassValue(classValue);
        // CLASSVALUE OF FIRST INSTANCE
        Evaluation evalj48 = new Evaluation(data);
        // NUMFOLDS 5 AND RANDOM 20
        int numFoldsj48=9;
        evalj48.crossValidateModel(tree,data,numFoldsj48,new Random(1));
        //FMEASURE RECALL AND PRECISION
        
        System.out.println(evalj48.toSummaryString("\nresults in forest",true));
        System.out.println((evalj48.kappa())+ "Kappa statistic for gaussian random forest is");
        
    }
    
}
