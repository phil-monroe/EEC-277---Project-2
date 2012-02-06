echo "AREA TEST ---------------------------------------------------------------"
./wesBench-instructional -doareatest

echo "LIGHT TEST --------------------------------------------------------------"
./wesBench-instructional -light

echo "TEXTURE TEST ------------------------------------------------------------"
./wesBench-instructional -tx 128

echo "TEXTURE AND LIGHT TEST --------------------------------------------------"
./wesBench-instructional -tx 128 -light

echo "DISJOINT TEST -----------------------------------------------------------"
./wesBench-instructional -tt 0

echo "TSTRIP TEST -------------------------------------------------------------"
./wesBench-instructional -tt 1

echo "INDEXED DISJOINT TEST ---------------------------------------------------"
./wesBench-instructional -tt 2

echo "INDEXED TSTRIP TEST -----------------------------------------------------"
./wesBench-instructional -tt 3

echo "VERTEX TEST -------------------------------------------------------------"
./wesBench-instructional -dovbtest

echo "TEXTURE TEST ------------------------------------------------------------"
./wesBench-instructional -dotxtest





