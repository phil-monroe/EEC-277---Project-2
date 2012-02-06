def parseAreaTest(testNum, filename, dest):
    f = open(filename, 'r');
    d = open(dest, 'w');
    a = 0;
    testOn = 0;
    while(True):
        if (a < 3):
            temp = f.readline();
            if (temp.find("----------------------------")>-1):
                a = a+1;
        else:
            testOn = testOn + 1;   
            if (testOn < testNum):
                a = 0;
            else:
                temp = f.readline();
                if ((len(temp) == 0) or (temp.find("Vendor:")>=0)):
                    d.close();
                    f.close();
                    return
                if (temp.find("area=") > -1):
                    tempArea = temp[int(temp.find("area=")+5): int(temp.find(" px,"))];
                    tempTris = temp[temp.find("tri rate = ")+11: temp.find(" Mtri/sec,")];
                    tempVerts = temp[temp.find("vertex rate=")+12: temp.find(" Mverts/sec,")];
                    tempFill = temp[temp.find("fill rate = ")+12: temp.find(" Mpix/sec")];
                    d.write(tempArea + "," + tempTris + "," +tempVerts + "," + tempFill  + "\n");
    
