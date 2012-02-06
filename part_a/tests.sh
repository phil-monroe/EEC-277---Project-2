# echo "AREA TEST ---------------------------------------------------------------"
# ./wesBench-instructional -doareatest

echo "LIGHT TEST --------------------------------------------------------------"
./wesBench-instructional -light -doareatest

echo "TEXTURE TEST ------------------------------------------------------------"
./wesBench-instructional -tx 128 -doareatest

echo "TEXTURE AND LIGHT TEST --------------------------------------------------"
./wesBench-instructional -tx 128 -light -doareatest

echo "DISJOINT TEST -----------------------------------------------------------"
./wesBench-instructional -tt 0 -doareatest

echo "TSTRIP TEST -------------------------------------------------------------"
./wesBench-instructional -tt 1 -doareatest

echo "INDEXED DISJOINT TEST ---------------------------------------------------"
./wesBench-instructional -tt 2 -doareatest

echo "INDEXED TSTRIP TEST -----------------------------------------------------"
./wesBench-instructional -tt 3 -doareatest

# echo "VERTEX TEST -------------------------------------------------------------"
# ./wesBench-instructional -dovbtest
# 
# echo "TEXTURE TEST ------------------------------------------------------------"
# ./wesBench-instructional -dotxtest





