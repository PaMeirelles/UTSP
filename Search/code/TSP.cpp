#include "include/TSP_IO.h"
#include "include/TSP_Basic_Functions.h"
#include "include/TSP_Init.h"
#include "include/TSP_2Opt.h"
#include "include/TSP_MCTS.h"
#include "include/TSP_Markov_Decision.h"
#include "include/TSP_sym.h"
#include <algorithm> // For std::find

// For TSP20-50-100 instances
void Solve_One_Instance(int Inst_Index)
{	
	Current_Instance_Begin_Time=(double)clock();  
	Current_Instance_Best_Distance=Inf_Cost;   	   
	
	// Input			
    cout << "Start Fetch" << endl;
	Fetch_Stored_Instance_Info(Inst_Index);	
	
    cout << "Start Preprocess" << endl;
	//Pre-processing	
	Calculate_All_Pair_Distance();	 	
  	Identify_Candidate_Set();    
	  
    cout << "Start MDP Search" << endl;
	//Search by MDP  	 		  		    
	Markov_Decision_Process(Inst_Index);
			
	double Stored_Solution_Double_Distance=Get_Stored_Solution_Double_Distance(Inst_Index);
	double Current_Solution_Double_Distance=Get_Current_Solution_Double_Distance();
			
	if(Stored_Solution_Double_Distance/Magnify_Rate-Current_Solution_Double_Distance/Magnify_Rate > 0.000001)
		Beat_Best_Known_Times++;	
	else if(Current_Solution_Double_Distance/Magnify_Rate-Stored_Solution_Double_Distance/Magnify_Rate > 0.000001)
		Miss_Best_Known_Times++;
	else
		Match_Best_Known_Times++;	
			
	Sum_Opt_Distance+=Stored_Solution_Double_Distance/Magnify_Rate;
	Sum_My_Distance+=Current_Solution_Double_Distance/Magnify_Rate;	
	Sum_Gap += (Current_Solution_Double_Distance-Stored_Solution_Double_Distance)/Stored_Solution_Double_Distance;
		
	printf("\nInst_Index:%d Concorde Distance:%f, MCTS Distance:%f Improve:%f Time:%.2f Seconds\n", Inst_Index+1, Stored_Solution_Double_Distance/Magnify_Rate, 
			Current_Solution_Double_Distance/Magnify_Rate, Stored_Solution_Double_Distance/Magnify_Rate-Current_Solution_Double_Distance/Magnify_Rate, ((double)clock()-Current_Instance_Begin_Time)/CLOCKS_PER_SEC);
			
	FILE *fp;   
	fp=fopen(Statistics_File_Name, "a+");     
	fprintf(fp,"\nInst_Index:%d \t City_Num:%d \t Concorde:%f \t MCTS:%f Improve:%f \t Time:%.2f Seconds\n",Inst_Index+1, Virtual_City_Num, Stored_Solution_Double_Distance/1000000,
			Current_Solution_Double_Distance/Magnify_Rate, Stored_Solution_Double_Distance/Magnify_Rate-Current_Solution_Double_Distance/Magnify_Rate, ((double)clock()-Current_Instance_Begin_Time)/CLOCKS_PER_SEC); 
	
	fprintf(fp,"Solution: ");
	int Cur_City=Start_City;
	do
	{
		fprintf(fp,"%d ",Cur_City+1);
		Cur_City=All_Node[Cur_City].Next_City;		
	}while(Cur_City != Null && Cur_City != Start_City);
	
	fprintf(fp,"\n"); 
	fclose(fp); 	
			
	Release_Memory(Virtual_City_Num);	
}
 
bool Solve_Instances_In_Batch()
{ 
	ifstream FIC;
	FIC.open(Input_File_Name);  
  
	if(FIC.fail())
	{
    	cout << "\n\nError! Fail to open file"<<Input_File_Name<<endl;
    	return false;
	}
  	else
    	cout << "\n\nBegin to read instances information from "<<Input_File_Name<<endl;


 	double Temp_X;
 	double Temp_Y;
 	int Temp_City;
    int Rec_Index;
    double Rec_Value;
 	char Temp_String[100];

    // MODIFIED: Force processing of exactly 1 instance, regardless of file headers
    int Process_Num = 1;
    cout << "Forced Processing of " << Process_Num << " instance(s)." << endl;

  	for(int i=0; i<Process_Num; i++)
  	{
  		for(int j=0;j<Temp_City_Num;j++)
  		{
			FIC>>Temp_X;
			FIC>>Temp_Y;
			Stored_Coordinates_X[i][j]=Temp_X;
			Stored_Coordinates_Y[i][j]=Temp_Y;
		}

		FIC>>&Temp_String[0];
		for(int j=0;j<Temp_City_Num;j++)
  		{
			FIC>>Temp_City;
			Stored_Opt_Solution[i][j]=Temp_City-1;
		}

		FIC>>Temp_City;

        FIC >> &Temp_String[0];
        for (int j = 0; j < Temp_City_Num; ++j) {
            for (int k = 0; k < Rec_Num; ++k) {
                FIC >> Rec_Index;
                Sparse_Stored_Rec[i][j].push_back(Rec_Index - 1);
            }
        }
        FIC >> &Temp_String[0];
        for (int j = 0; j < Temp_City_Num; ++j) {
            for (int k = 0; k < Rec_Num; ++k) {
                FIC >> Rec_Value;
                Sparse_Stored_Rec_Value[i][j].push_back(Rec_Value);
            }
        }

        for (int j = 0; j < Temp_City_Num; ++j) {
		    for(int k = 0; k < Temp_City_Num; ++k)
		    {
		        Stored_Rec_Value[i][j].push_back(0.0);
		    }
        }

    	for (int j = 0; j < Temp_City_Num; ++j) {
    	    for (int l = 0; l < Temp_City_Num; ++l) {
    	        auto it = std::find(Sparse_Stored_Rec[i][j].begin(), Sparse_Stored_Rec[i][j].end(), l);
    	        if (it != Sparse_Stored_Rec[i][j].end()) {
    	            int index = std::distance(Sparse_Stored_Rec[i][j].begin(), it);
    		        Stored_Rec_Value[i][j][l] =  Sparse_Stored_Rec_Value[i][j][index];
    	        }
    	    }
    	}

    	// H' = H + H^T
    	symmetrizeMatrix(Stored_Rec_Value[i], Max_City_Num);

    	for (int j = 0; j < Temp_City_Num; ++j) {
    		for (int m = 0; m < Temp_City_Num; ++m)
                   	Stored_Rec[i][j].push_back(m);
    	}
	}
  	FIC.close();

    // MODIFIED: Ignore batch math, just solve index 0
	Test_Inst_Num = 1;

	FILE *fp;
	fp=fopen(Statistics_File_Name, "w+");
	fprintf(fp,"Number_of_Instances_In_Current_Batch: %d\n",Test_Inst_Num);
	fclose(fp);

	// Just solve the first one (index 0)
    Solve_One_Instance(0);

  	return true;
}

int main(int argc, char ** argv)
{
	double Overall_Begin_Time=(double)clock();

    srand(time(NULL));

    // MODIFIED: Arguments logic kept same, but Index_In_Batch will effectively be ignored by our new logic
	Index_In_Batch=atoi(argv[1]);
	Statistics_File_Name=argv[2];
	Input_File_Name=argv[3];
	Temp_City_Num=atoi(argv[4]);
    use_rec = atoi(argv[5]);
    rec_only = atoi(argv[6]);

    Max_Candidate_Num = atoi(argv[7]);
    Max_Depth = atoi(argv[8]);
    Alpha = atof(argv[9]);
    Beta = atof(argv[10]);
    Param_H = atof(argv[11]);
    restart = atoi(argv[12]);
    restart_reconly = atoi(argv[13]);

    cout << "record some exp parameters here: !!" << endl;
    cout << "Alpha: " << Alpha << endl;
    cout << "Beta: " << Beta << endl;
    cout << "Param_H: " << Param_H << endl;
    cout << "Param_T: " << Param_T << endl;
    cout << "#Candidate Set: " << Max_Candidate_Num << endl;
    cout << "Max Depth " << Max_Depth << endl;
    cout << "rec_only " << rec_only << endl;
    cout << "restart" << restart << endl;
    cout << "restart_reconly" << restart_reconly << endl;

	Solve_Instances_In_Batch();

	FILE *fp;
	fp=fopen(Statistics_File_Name, "a+");
	fprintf(fp,"\n\nIndex_In_Batch: %d, Avg_Concorde_Distance: %f Avg_MCTS_Distance: %f Avg_Gap: %f Total_Time: %.2f Seconds \n Beat_Best_Known_Times: %d Match_Best_Known_Times: %d Miss_Best_Known_Times: %d \n",
			Index_In_Batch, Sum_Opt_Distance/Test_Inst_Num,Sum_My_Distance/Test_Inst_Num, Sum_Gap/Test_Inst_Num, ((double)clock()-Overall_Begin_Time)/CLOCKS_PER_SEC, Beat_Best_Known_Times, Match_Best_Known_Times, Miss_Best_Known_Times);
	fclose(fp);

	printf("\n\nIndex_In_Batch: %d, Avg_Concorde_Distance: %f Avg_MCTS_Distance: %f Avg_Gap: %f Total_Time: %.2f Seconds \n Beat_Best_Known_Times: %d Match_Best_Known_Times: %d Miss_Best_Known_Times: %d \n",
			Index_In_Batch, Sum_Opt_Distance/Test_Inst_Num,Sum_My_Distance/Test_Inst_Num, Sum_Gap/Test_Inst_Num, ((double)clock()-Overall_Begin_Time)/CLOCKS_PER_SEC, Beat_Best_Known_Times, Match_Best_Known_Times, Miss_Best_Known_Times);

    // MODIFIED: REMOVED getchar() TO PREVENT HANGING
	// getchar();

	return 0;
}