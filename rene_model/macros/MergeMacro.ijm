// macro to make merged image out of 4 image files in a folder where the 
// image names are name-1_blue.png, ..., name-x_yellow.png
// saves a list of processed images in the end

dir1=getDirectory("choose input directory");
dir2=getDirectory("create or choose output directory");
dir3=getDirectory("select directory for picture name list");

channels = newArray(4);

startingValue = 0
endValue = 3

list = getFileList(dir1);
print("Id, Guess")

//Set the size of the loop
listLength = list.length/4;
for (i = 0; i < listLength; i++)
{
// Open all images
	for (j = startingValue; j <= endValue; j++)
	{ 
		open(dir1 + list[j]);
		name=getInfo("image.filename");               	
		
	}
	// Assign image names to channels
	for (k = startingValue; k <= endValue; k++)
	{	
		if (indexOf(list[k], "blue") >= 0)
		{
			channels[0] = list[k];
		}
		else if (indexOf(list[k], "green") >= 0)
		{
			channels[1] = list[k];		
		}
		else if (indexOf(list[k], "red") >= 0)
		{
			channels[2] = list[k];	
		}
		else if (indexOf(list[k], "yellow") >= 0)
		{
			channels[3] = list[k];		
		}
	}
	

	// Merge channels to one 4 channel image
	run("Merge Channels...", "c1="+channels[2]+" c2="+channels[1]+" c3="+channels[0]+" c7="+channels[3]+" create");

	//splits image name at "_" 
	nameArray = split(name, "_");
	saveName = nameArray[0];
	print(saveName + ",");
	
	// save ...
	saveAs("Tiff", dir2+saveName);
	close();

	
	//loop starts merging the next four files
	startingValue = startingValue + 4;
	endValue = endValue + 4;
}
//save Log window as text file
selectWindow("Log");
saveAs("Text", dir3 + "/200_names.txt");
print("merging completed");
print("clear Log window");
print("Run GuessMacro");