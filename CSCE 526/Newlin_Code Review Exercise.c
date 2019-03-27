/******************************************************************************

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, PHP, Ruby, 
C#, VB, Perl, Swift, Prolog, Javascript, Pascal, HTML, CSS, JS
Code, Compile, Run and Debug online from anywhere in world.

*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/types.h>

/**
 * Author: Marvin Newlin
 * This Function takes in a mode and two directory names
 * Creates some sample files and places them in the two directories
 * created (error checked). It then performs the mode on the 2 directories
 * 
 */


/** 
 *  This function syncs the contents of the 2 directories
 *  Sync in this case means that the file names are the same.
 */
void sync (DIR *directory1, char *dir1, DIR *directory2, char *dir2)
{
    printf("Sync\n");
    int inList = 0; //boolean for add or not
    //Check contents of 1st dir against second dir and add missing items
    struct dirent *dp1;
    struct dirent *dp2;
    if (directory1 != NULL && directory2 != NULL) {
        while (dp1 = readdir(directory1)) {
            inList = 0;
            if (strcmp(dp1->d_name, ".")!= 0 && strcmp(dp1->d_name, "..") != 0) {
                while (dp2 = readdir(directory2)) {
                    if (strcmp(dp1->d_name, dp2->d_name) == 0) {
                        inList = 1;
                        break;
                    }
                }
                if (inList == 0) {
                    char cmdString[40];
                    sprintf(cmdString, "echo > %s/%s", dir2, dp1->d_name);
                    system(cmdString);
                }
            }
        }
        directory1 = opendir(dir1);
        directory2 = opendir(dir2);
        inList = 0;
        while (dp2 = readdir(directory2)) {
            inList = 0;
            if (strcmp(dp2->d_name, ".")!= 0 && strcmp(dp2->d_name, "..") != 0) {
                while (dp1 = readdir(directory1)) {
                    if(strcmp(dp2->d_name, dp1->d_name) == 0) {
                        inList = 1;
                        break;
                    }
                }
                if (inList == 0) {
                    char cmdString[40];
                    sprintf(cmdString, "echo > %s/%s", dir1, dp2->d_name);
                    system(cmdString);
                }
            }
        }
    }
    
}

/*
    This function de-duplicates the contents of the two directories
    De-duplicate means that none of the filenames are the same.
*/
void deduplicate(DIR *directory1, char *dir1, DIR *directory2, char *dir2)
{
    printf("Deduplicate\n");
    //Check contents of 1st dir against second dir and remove dupolicate items
    struct dirent *dp1;
    struct dirent *dp2;
    directory1 = opendir(dir1);
    
    if (directory1 != NULL && directory2 != NULL) {
        dp1 = readdir(directory1);
        while (dp1) {
            directory2 = opendir(dir2);
            if (directory2 != NULL) {
                dp2 = readdir(directory2);
                while (dp2) {
                    if (strcmp(dp1->d_name, dp2->d_name) == 0) {
                        if (strcmp(dp1->d_name, ".")!= 0 && strcmp(dp1->d_name, "..") != 0) {
                            char cmdString[40];
                            sprintf(cmdString, "rm %s/%s", dir1, dp1->d_name);
                            system(cmdString);
                        }
                    }
                    dp2 = readdir(directory2);
                }
                dp1 = readdir(directory1);
            }
        }
    }
}

/**
 * Valid file names are file names with alpha numeric characters,/, -, and _
 * All other characters in a file name will result in an invalid file name
 * for this program
 */
void checkFileName(char *fileName, int len) {
    if (strlen(fileName) <= len) {
      len = strlen(fileName);
    }
    for (int i = 0; i < len; i++) {
       if (isalnum (fileName[i]) == 0) {
    	  if (fileName[i] != '-' && fileName[i] != '_' && fileName[i] != '/') {
    	      exit (EXIT_FAILURE);
    	  }
      }
    }
}

/**
 * This program takes in the mode, Sync or duplicate (S or D)
 * and two directory names. The program validates these file names and then
 * creates directories with these names. The program will then generate test
 * files to put into the folders and then run the operation on the two folders
 * 
 */
int main (int argc, char *argv[]) {
    
  if (argc != 4) {
      printf ("Invalid parameters\n");
      exit (EXIT_FAILURE);
  }
  char mode = argv[1][0];
  if (mode != 'S' && mode != 's' && mode != 'D' && mode != 'd') {
      printf ("Invalid parameters\n");
      exit (EXIT_FAILURE);
  }
    
    
  char dir1[26];
  char dir2[26];
  
  //Check for bad file names
  int len = 25;
  char temp1[26];
  char temp2[26];
  if (sizeof(argv[2]) > 23 && sizeof(argv[3]) > 23) {
      exit(EXIT_FAILURE);
  }
  
  strcpy(temp1, argv[2]);
  strcpy(temp2, argv[3]);
  
  checkFileName(temp1, len);
  checkFileName(temp2, len);
  //Protecting against a buffer overrun here
  if(sizeof(argv[2]) < 23 && sizeof(argv[3]) < 23) {
      sprintf (dir1, "./%s", argv[2]);
      sprintf (dir2, "./%s", argv[3]);
  } else {
      exit(EXIT_FAILURE);
  }
  
  dir1[sizeof(dir1)-1] = '\0';
  dir2[sizeof(dir2)-1] = '\0';
    
  int op_mode;
  if (mode == 'S' || mode == 's') {
      op_mode = 1;
  } else {
      op_mode = 0;
  }
    
  
   //If we make it here then we have validated the file names syntactically
  //Time to run our operations

  //Create directories
  int op1 = mkdir (dir1);
  int op2 = mkdir (dir2);
  if (op1 != 0 || op2 != 0)
  {
      //Something has happened that prevents us from creating directories
      exit(EXIT_FAILURE);
  }
 
  DIR *directory1 = opendir (dir1);
  DIR *directory2 = opendir (dir2);

  //Generate Test files
  char testFileName1[] = "test1.txt";
  char testFileName2[] = "test2.txt";
  char testFileName3[] = "test3.txt";
  char testFileName4[] = "test4.txt";

  char path1[50];
  char path2[50];
  char path3[50];
  char path4[50];
  char path5[50];
  char path6[50];
  snprintf (path1, 49, "%s/%s", dir1, testFileName1);
  snprintf (path2, 49, "%s/%s", dir2, testFileName2);
  snprintf (path3, 49, "%s/%s", dir1, testFileName2);
  snprintf (path4, 49, "%s/%s", dir2, testFileName3);
  snprintf (path5, 49, "%s/%s", dir1, testFileName3);
  snprintf (path6, 49, "%s/%s", dir2, testFileName4);
  
  path1[strlen(path1)] = '\0';
  path2[strlen(path2)] = '\0';
  path3[strlen(path3)] = '\0';
  path4[strlen(path4)] = '\0';
  path3[strlen(path5)] = '\0';
  path4[strlen(path6)] = '\0';
  
  //Actually create files
  FILE *f1 = fopen (path1, "w+");
  FILE *f2 = fopen (path2, "w+");
  FILE *f3 = fopen (path3, "w+");
  FILE *f4 = fopen (path4, "w+");
  FILE *f5 = fopen (path5, "w+");
  FILE *f6 = fopen (path6, "w+");
  //Close files so we can manipulate them
  fclose(f1);
  fclose(f2);
  fclose(f3);
  fclose(f4);
  fclose(f5);
  fclose(f6);
  
  char temp[30];
  printf("Initial directory contents\n");
  sprintf(temp, "ls %s", dir1);
  system(temp);
  sprintf(temp, "ls %s", dir2);
  system(temp);
  
  if (op_mode == 1) {
      sync(directory1, dir1, directory2, dir2);
  } 
  if (op_mode == 0) {
      deduplicate(directory1, dir1, directory2, dir2);
  }
  
  printf("Post-operation directory contents\n");
  sprintf(temp, "ls %s", dir1);
  system(temp);
  sprintf(temp, "ls %s", dir2);
  system(temp);
  exit (EXIT_SUCCESS);
}




