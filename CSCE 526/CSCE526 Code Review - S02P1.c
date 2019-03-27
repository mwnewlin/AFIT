******************************************************************************

6 November 2016
Student 2
CSCE 526

*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
#define BUF_SIZE ((int) 32)

/*Prints usage statement if error occurs or input is invalid*/
void usage(){
printf("USAGE: filer [s | d] [directory 1] [directory 2] with no slashes\n\
s - sync mode, copy all uniquely named files from directory 1 to + directory 2 \
and vice versa.\nd - Deduplicate mode, delete all files + with same name from \
directory 2.\nDirectories should not include slashes");
}

// Creates files to ensure the folders have files for manipulation.
void createFiles(){
system("mkdir dir1");
system("echo \"this is file1\" > dir1/file1");
system("echo \"this is file2\" > dir1/file2");
system("mkdir dir2");
system("echo \"this is file2\" > dir2/file2");
system("echo \"this is file3\" > dir2/file3");
}


//Check that the user-inputted directories exist, return 0 if non-existent
int checkDirs(char fp1[], char fp2[]){
DIR *dip;
if((isalpha(fp1[0]) || isdigit(fp1[0])) && (isalpha(fp2[0]) || isdigit(fp2[0]))){
if ((dip = opendir(fp1)) != NULL) { 
if((dip = opendir(fp2)) != NULL) {
closedir(dip);
return 0;
} 
else {
closedir(dip);
return 1;
}
} else { return 1; }
} else { return 1; }
}


// Counts the numebr of files in a given directory
int countFiles(char fp[]){
DIR *dip1 = opendir(fp);
struct dirent *dir;
int file_count = 0;

while ((dir = readdir(dip1)) != NULL)
{
if(strcmp(dir->d_name,".") && strcmp(dir->d_name,"..")){ file_count++; } 
}
closedir(dip1);
return file_count;
}

/*Inputs the names of files in the directory fp1 and puts them into memory at 
filenames1*/
void listFiles(char *filenames[], char fp1[]){
DIR *dip1 = opendir(fp1);
struct dirent *dir;
int i = 0;

while ((dir = readdir(dip1)) != NULL){
if(!strcmp(dir->d_name, ".") || !strcmp(dir->d_name, "..")) {} 
else {
filenames[i] = (char*) malloc(strlen(dir->d_name)+1);
strncpy(filenames[i], dir->d_name, strlen(dir->d_name));
i++;
}
}
return;
}


/* For this program, sync means that the folder names will be the same, but the
contents may be different. If a folder does not have a specific file, it will
be copied from the other, but if a file exists in each with the same name, 
the contents will not be checked. */
int sync(char fp1[], char fp2[]){
int file_count1 = countFiles(fp1);
const char *files1[file_count1];
listFiles(files1, fp1);

int file_count2 = countFiles(fp2);
const char *files2[file_count2];
listFiles(files2, fp2);

for(int i=0; i<file_count1; i++){
int match = 0;
for(int j=0; j<file_count2; j++){
if(strcmp(files1[i], files2[j]) == 0){ match = 1; }
}
//Copy files1[i] into fp2
if(match == 0){
char buffer[BUF_SIZE];
snprintf(buffer, BUF_SIZE, "cp %s/%s %s/%s%c", fp1, files1[i], fp2, files1[i],'\0');
system(buffer);
}
}
for(int i=0; i<file_count2; i++){
int match = 0;
for(int j=0; j<file_count1; j++){
if(strcmp(files2[i], files1[j]) == 0){ match = 1; }
}
//Copy files2[i] into fp1
if(match == 0){
char buffer[BUF_SIZE];
snprintf(buffer, BUF_SIZE, "cp %s/%s %s/%s%c", fp2, files2[i], fp1, files2[i],'\0');
system(buffer);
}
}
return 0;
}


/*For this program, deduplicate means the program will account for all files
in the first directory, and if there are any by the same name in the second
directory, those files will be deleted from the second directory.*/
int deduplicate(char fp1[], char fp2[]){
int file_count1 = countFiles(fp1);
const char *files1[file_count1];
listFiles(files1, fp1);

int file_count2 = countFiles(fp2);
const char *files2[file_count2];
listFiles(files2, fp2);

for(int i=0; i<file_count1; i++){
int match = 0;
for(int j=0; j<file_count2; j++){
if(strcmp(files1[i], files2[j]) == 0){ match = 1; }
}
//Remove duplicate files from the second directory
if(match == 1){
char buffer[BUF_SIZE];
snprintf(buffer, BUF_SIZE, "rm %s/%s%c", fp2, files1[i],'\0');
printf("%s",buffer);
system(buffer);
}
}
return 0;
}

int main( int argc, char *argv[] )
{
createFiles();
system("ls dir1\n");
system("ls dir2\n");
if( argc != 4 ) { usage(); } 
else{
char filepath1[BUF_SIZE];
char filepath2[BUF_SIZE];
snprintf(filepath1, BUF_SIZE-1, "%s%c", argv[2], '\0');
snprintf(filepath2, BUF_SIZE-1, "%s%c", argv[3], '\0');
if(checkDirs(&filepath1, &filepath2) == 0){
if(strcmp(argv[1], "d") == 0){ deduplicate(filepath1, filepath2); }
else if(strcmp(argv[1], "s") == 0) { sync(filepath1, filepath2); }
else{ usage(); }
} else { usage(); }
}
system("ls dir1\n");
system("ls dir2\n");
return 0;
}

 

 