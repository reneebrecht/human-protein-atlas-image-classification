// Macro to open images and make a prediction
// saves a list of processed images in the end 

dir2=getDirectory("Choose input directory merged images");
dir3=getDirectory("select folder to save names of processed images");

print("processed image");

list = getFileList(dir2);
listLength = list.length;

for (i = 0; i < listLength; i++)
{
    open(dir2 + list[i]);
	name=getInfo("image.filename"); 
	print(name);
    run("Channels Tool...");
	Stack.setActiveChannels("0100");
	run("Brightness/Contrast...");
	waitForUser("guess and OK");
	close();
}
//save Log window as text file
selectWindow("Log");
saveAs("Text", dir3 + "/200_names_2.txt");	
print("rene model completed predictions");
print("please save your prediction file as a csv");