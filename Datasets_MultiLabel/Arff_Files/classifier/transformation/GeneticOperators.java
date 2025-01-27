package mulan.classifier.transformation;
import java.util.Random;

import mulan.classifier.InvalidDataException;

/**
 * Class for using Genetic Operators for label ordering in CC
 * @author Serafin
 *
 */

public class GeneticOperators {
	/**
	 * It generates a random permutation
	 * @param num_labels the number of labels
	 * @return the array with the permutation
 */
	
	public static int[] generateRandomSolution(int num_labels) {
		int[] permutation = new int[num_labels];
		int remaining_labels = num_labels;
		Random random = new Random();
		int random_index;
		int temp;
		
		// At the beginning, 1,2,,,,num_labels
		
		for(int i = 0; i < num_labels; i++) {
			permutation[i] = i;
		}
		
		// Select a random label and change the positions with permutation[i]
		
		for(int i = num_labels - 1; i >= 0; i--) {
			random_index = random.nextInt(remaining_labels);
			temp = permutation[random_index];
			permutation[random_index] = permutation[i];
			permutation[i] = temp;
			remaining_labels--;
		}
		
		return permutation;
	}
	/**
	 * Cross two parents
	 * select a random sub-chain and a random position where insert it
	 * Insert the random sub-chain of the donor in the child
	 * Insert the rest of the elements of the receptor preserving the order 
	 * @param donor the donor
	 * @param receptor the receptor
	 * @return the generated child
	 */
	
	public static int[] crossOperator(int[] donor, int[] receptor) {
		int num_labels = donor.length;
		int[] child = new int[num_labels];
		int random_position1, random_position2;
		int length;
		int random_position_insertion;
		int position_child, position_donor;
		Random random = new Random();
		int temp;
		boolean already_inserted;
		int k;
		
		random_position1 = random.nextInt(num_labels);
		random_position2 = random.nextInt(num_labels);
		
		// Force that random_position1 <= random_position2
		
		if(random_position1 > random_position2) {
			temp = random_position1;
			random_position1 = random_position2;
			random_position2 = temp;
		}
		
		length = random_position2 - random_position1 + 1;
		
		/* Select the position where insert the sub-chain
		 There must be enough space to insert the sub-chain*/
		
		random_position_insertion = random.nextInt(num_labels-length+1);
		
		// Copy the sub-chain of the donor in the child
		
		for(int i = 0; i < length; i++) {
			position_child = random_position_insertion+i;
			position_donor = random_position1 + i;
			child[position_child] = donor[position_donor];
		}
		
		// Copy the resting sub-chain following the order of the receptor 
		
		if(random_position_insertion > 0)
			position_child = 0;
		
		else
			position_child = length;
		
		for(int i = 0; i < num_labels; i++) {
			/* For each element in the receptor, check if it belongs 
			to the sub-chain of the donor */
			
			already_inserted = false;
			k = random_position1;
			
			while(k <= random_position2 && !already_inserted) {
				if(donor[k] == receptor[i])
					already_inserted = true;
				
				else
					k++;
			}
			/*
			 * Copy in the child if it does not belong to the chain of the receptor
			 * Update the position of inserting of the child 
			 * Be careful with not enter in the subchain of the donor
			 */
			if(!already_inserted) {
				child[position_child] = receptor[i];
				position_child++;
			
				if(position_child == random_position_insertion)
					position_child += length;
			}
		}
		
		return child;
	}
	
	/**
	 * Creates a permutation mutated of the first one
	 * The mutation is created by changing two random components
	 * @param initial_permutation the initial permutation 
	 * @return The mutated permutation 
	 */
	
	public static int[] mutatePermutation(int[] initial_permutation){
		int random_position1, random_position2;
		Random random = new Random();
		int num_labels = initial_permutation.length;
		int[] mutated_permutation = new int[num_labels];
		int temp;
		
		//Initially, the mutated permutation is equal to the initial one
		
		for(int i = 0; i < num_labels; i++)
			mutated_permutation[i] = initial_permutation[i];
		
		// Select two random components
		
		random_position1 = random.nextInt(num_labels);
		random_position2 = random.nextInt(num_labels);

		temp = mutated_permutation[random_position1];
		mutated_permutation[random_position1] = mutated_permutation[random_position2];
		mutated_permutation[random_position2] = temp;
		
		return mutated_permutation;
	}
	
	
	
	public static void main(String[] args) throws InvalidDataException, Exception {
		int[] permutation = generateRandomSolution(6);
		int[] mutated_permutation = mutatePermutation(permutation);
		int[] donor = {6,2,3,4,1,5};
		int[] receptor = {1,4,6,3,5,2};
		int[] child = crossOperator(donor, receptor);
		System.out.println("Fin del chequeo");
	}
	
	

}
