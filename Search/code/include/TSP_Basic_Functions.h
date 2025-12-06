
// Return an integer between [0,Divide_Num)
int Get_Random_Int(int Divide_Num)
{ 
  return rand()%Divide_Num;
}


double Calculate_Att_Distance(int First_City,int Second_City)
{
	double Rij = sqrt( (Coordinate_X[First_City]-Coordinate_X[Second_City])*(Coordinate_X[First_City]-Coordinate_X[Second_City]) +
				   (Coordinate_Y[First_City]-Coordinate_Y[Second_City])*(Coordinate_Y[First_City]-Coordinate_Y[Second_City]) ) /10.0;
  
  	int Tij = (int)(Rij + 0.5);
  
  	if(Tij < Rij)
		return Tij + 1;
  	else
		return Tij;
}

int Calculate_Int_Att_Distance(int First_City,int Second_City)
{
	return (int)(0.5 + Calculate_Att_Distance(First_City,Second_City));
}


//Calculate the geo distance between two cities, rounded to the nearest integer 
double Calculate_Geo_Distance(int First_City,int Second_City)
{
	double RR = 6371;

	double PI = 3.141592;

	double Latitude_X1 = Coordinate_X[First_City]*PI/180.0;
	double Latitude_X2 = Coordinate_X[Second_City]*PI/180.0;
	double Longitude_Y1 = Coordinate_Y[First_City]*PI/180.0;
	double Longitude_Y2 = Coordinate_Y[Second_City]*PI/180.0;

	double a = pow(sin((Latitude_X2 - Latitude_X1)/2), 2) + 
		cos(Latitude_X1)*cos(Latitude_X2) * 
		pow(sin((Longitude_Y2 - Longitude_Y1)/2),2);

	double c = 2 * atan2(sqrt(a), sqrt(1-a));

	return RR*c;
}

int Calculate_Int_Geo_Distance(int First_City,int Second_City)
{
	return (int)(0.5 + Calculate_Geo_Distance(First_City,Second_City));
}


//Calculate the distance between two cities, rounded up to the nearest integer 
int Calculate_Int_Distance(int First_City,int Second_City)
{
  	return (int)(0.5 + sqrt( (Coordinate_X[First_City]-Coordinate_X[Second_City])*(Coordinate_X[First_City]-Coordinate_X[Second_City]) +
                (Coordinate_Y[First_City]-Coordinate_Y[Second_City])*(Coordinate_Y[First_City]-Coordinate_Y[Second_City]) ) );
}

//Calculate the distance between two cities
double Calculate_Double_Distance(int First_City,int Second_City)
{
  	return sqrt( (Coordinate_X[First_City]-Coordinate_X[Second_City])*(Coordinate_X[First_City]-Coordinate_X[Second_City]) +
                   (Coordinate_Y[First_City]-Coordinate_Y[Second_City])*(Coordinate_Y[First_City]-Coordinate_Y[Second_City]) );
}


// Fetch the distance (already stored in Distance[][]) between two cities 
Distance_Type Get_Distance(int First_City,int Second_City)
{
	return Distance[First_City][Second_City];
}

// Using the information stored in Solution[] to update the information stored in Struct_Node *All_Node
void Convert_Solution_To_All_Node()
{
  	int Temp_Cur_City;
  	int Temp_Pre_City;
  	int Temp_Next_City;
  	int Cur_Salesman=0;
  
  	for(int i=0;i<Virtual_City_Num;i++)
  	{
  		Temp_Cur_City = Solution[i];  	
  		Temp_Pre_City = Solution [(i-1+Virtual_City_Num)%Virtual_City_Num];
  		Temp_Next_City = Solution [(i+1+Virtual_City_Num)%Virtual_City_Num];
  	
  		if(Temp_Cur_City >= City_Num)
  	  		Cur_Salesman++;  	
  		
  		All_Node[Temp_Cur_City].Pre_City=Temp_Pre_City;
  		All_Node[Temp_Cur_City].Next_City=Temp_Next_City;  
		All_Node[Temp_Cur_City].Salesman=Cur_Salesman;   
  	} 
}

// Using the information stored in Struct_Node *All_Node to update the information stored in Solution[] 
bool Convert_All_Node_To_Solution()
{
	for(int i=0;i<Virtual_City_Num;i++)
		Solution[i]=Null;
	
	int Cur_Index=0;
	Solution[Cur_Index]=Start_City;
	
	int Cur_City=Start_City;
	do
	{
		Cur_Index++;
		
		Cur_City=All_Node[Cur_City].Next_City;
		if(Cur_City == Null || Cur_Index >= Virtual_City_Num)
			return false;
		
		Solution[Cur_Index]=Cur_City;			
	}while(All_Node[Cur_City].Next_City != Start_City);
	
	return true;
}

// Check the current solution stored in Struct_Node *All_Node is a feasible TSP tour
bool Check_Solution_Feasible()
{
	int Cur_City=Start_City;
	int Visited_City_Num=0;
	while(true)
	{	
		Cur_City = All_Node[Cur_City].Next_City;
		if(Cur_City == Null)	
		{
			printf("\nThe current solution is unvalid. Current city is Null\n");
			return false;
		}
		  		
		Visited_City_Num++;
		if(Visited_City_Num > Virtual_City_Num)
		{
			printf("\nThe current solution is unvalid. Loop may exist\n");
			getchar();
			return false;
		}
				
		if(Cur_City == Start_City && Visited_City_Num == Virtual_City_Num)	
			return true;			 	
	}
}

// Return the total distance (integer) of the solution stored in Struct_Node *All_Node
Distance_Type Get_Solution_Total_Distance()
{
  	Distance_Type Solution_Total_Distance=0;
  	for(int i=0;i<Virtual_City_Num;i++)
  	{
  		int Temp_Next_City=All_Node[i].Next_City;
  		if(Temp_Next_City != Null)
  	  		Solution_Total_Distance += Get_Distance(i,Temp_Next_City); 
  		else
  		{
  			printf("\nGet_Solution_Total_Distance() fail!\n");
  			getchar();
  			return Inf_Cost;
		}  	  		
  	}	
  
  	return Solution_Total_Distance;
}


// Return the unselected city neareast to Cur_City
int Get_Neareast_Unselected_City(int Cur_City)
{	
	int Neareast_Unselected_City=Null;
	for(int i=0;i<Virtual_City_Num;i++)
	{
		if (i==Cur_City || If_City_Selected[i] || Get_Distance(Cur_City,i) >= Inf_Cost || (rec_only && !recommend[Cur_City][i]))
			continue;
			
		if(Neareast_Unselected_City == Null || Get_Distance(Cur_City,i) < Get_Distance(Cur_City,Neareast_Unselected_City))
			Neareast_Unselected_City=i;
	}
	
	return Neareast_Unselected_City;
}

// Identify a set of candidate neighbors for each city, stored in Candidate_Num[] and Candidate[][]
void Identify_Candidate_Set()
{	
	for(int i=0;i<Virtual_City_Num;i++)
	{
		Candidate_Num[i]=0;
		
		for(int j=0;j<Virtual_City_Num;j++)
			If_City_Selected[j]=false;		
		
		for(int j=0;j<Max_Candidate_Num;j++)
		{
			int Neareast_Unselected_City=Get_Neareast_Unselected_City(i);
			if(Neareast_Unselected_City != Null)
			{
				Candidate[i][Candidate_Num[i]++]=Neareast_Unselected_City;		
				If_City_Selected[Neareast_Unselected_City]=true;
			}
	    }
	}
}

bool Check_If_Two_City_Same_Or_Adjacent(int First_City, int Second_City)
{
	if(First_City==Second_City || All_Node[First_City].Next_City == Second_City || All_Node[Second_City].Next_City == First_City)	
		return true;
	else
		return false;
}

// For each city between First_City and Second City, reverse its Pre_City and Next_City 
void Reverse_Sub_Path(int First_City,int Second_City)
{
	int Cur_City=First_City;
	int Temp_Next_City=All_Node[Cur_City].Next_City;
	
	while(true)
	{	
		int Temp_City = All_Node[Cur_City].Pre_City;
		All_Node[Cur_City].Pre_City=All_Node[Cur_City].Next_City;
		All_Node[Cur_City].Next_City=Temp_City;
		
		if(Cur_City==Second_City)
			break;
		
		Cur_City=Temp_Next_City;
		Temp_Next_City=All_Node[Cur_City].Next_City;	
	}	
} 

// Copy information from Struct_Node *All_Node to Struct_Node *Best_All_Node
void Store_Best_Solution()
{
	for(int i=0;i<Virtual_City_Num;i++)
	{
		Best_All_Node[i].Salesman=All_Node[i].Salesman;
		Best_All_Node[i].Next_City=All_Node[i].Next_City;
		Best_All_Node[i].Pre_City=All_Node[i].Pre_City;
	}	 
}

// Copy information from Struct_Node *Best_All_Node to Struct_Node *All_Node 
void Restore_Best_Solution()
{
	for(int i=0;i<Virtual_City_Num;i++)
	{
		All_Node[i].Salesman=Best_All_Node[i].Salesman;
		All_Node[i].Next_City=Best_All_Node[i].Next_City;
		All_Node[i].Pre_City=Best_All_Node[i].Pre_City;
	}	 
}



double Calculate_Distance(int First_City, int Second_City, int type, bool integer_distance)
{
	if(type == 0) {
		if(!integer_distance)
			return (int)(0.5 + Calculate_Double_Distance(First_City, Second_City));
		else
			return Calculate_Int_Distance(First_City, Second_City);
	} else if(type == 1) {
		if(!integer_distance)
			return (int)(0.5 + Calculate_Geo_Distance(First_City, Second_City));
		else
			return Calculate_Int_Geo_Distance(First_City, Second_City);
	} else if(type == 2) {
		if(!integer_distance)
			return (int)(0.5 + Calculate_Att_Distance(First_City, Second_City));
		else
			return Calculate_Int_Att_Distance(First_City, Second_City);

	}else
	{
		printf("\nCalculate_Distance(): wrong distance type!\n");
		getchar();
		return -1;
	}
}


// Calculate the distance (integer) between any two cities, stored in Distance[][]
void Calculate_All_Pair_Distance()
{
  	for(int i=0;i<Virtual_City_Num;i++)
  		for(int j=0;j<Virtual_City_Num;j++)
  		{
	  		if(i!=j)
	    		Distance[i][j]= Calculate_Distance(i,j,distance_type,true);// Calculate_Int_Distance(i,j);  
	  		else
	    		Distance[i][j]=Inf_Cost;  	  
		}  	 
}

//For TSP20-50-100 instances
// Return the total distance (double) of the solution stored in Stored_Opt_Solution[Inst_Index]
double Get_Stored_Solution_Double_Distance(int Inst_Index)
{
	double Stored_Solution_Double_Distance=0;
	for(int i=0;i<Virtual_City_Num-1;i++)
		Stored_Solution_Double_Distance += Calculate_Distance(Stored_Opt_Solution[Inst_Index][i],Stored_Opt_Solution[Inst_Index][i+1], distance_type, false); //Calculate_Double_Distance(Stored_Opt_Solution[Inst_Index][i],Stored_Opt_Solution[Inst_Index][i+1]);
	
	Stored_Solution_Double_Distance += Calculate_Distance(Stored_Opt_Solution[Inst_Index][Virtual_City_Num-1],Stored_Opt_Solution[Inst_Index][0], distance_type, false); //Calculate_Double_Distance(Stored_Opt_Solution[Inst_Index][Virtual_City_Num-1],Stored_Opt_Solution[Inst_Index][0]);	
	
	return Stored_Solution_Double_Distance;
}



//For TSP20-50-100 instances
// Return the total distance (double) of the solution stored in Struct_Node *All_Node
double Get_Current_Solution_Double_Distance()
{
  	double Current_Solution_Double_Distance=0;
  	for(int i=0;i<Virtual_City_Num;i++)
  	{
  		int Temp_Next_City=All_Node[i].Next_City;
  		if(Temp_Next_City != Null)
  	  		Current_Solution_Double_Distance += Calculate_Distance(i, Temp_Next_City, distance_type, false); //Calculate_Double_Distance(i,Temp_Next_City);
  		else
  		{
  			printf("\nGet_Current_Solution_Double_Distance() fail!\n");
  			getchar();
  			return Inf_Cost;
		}  	  		
  	}	
  
  	return Current_Solution_Double_Distance;
}
