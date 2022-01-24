# cylindrical-astar

## Please cite this work:
A. de Giorgio and L. Wang, **"Artificial Intelligence Control in 4D Cylindrical Space for Industrial Robotic Applications"**, in IEEE Access, vol. 8, pp. 174833-174844, 2020, [doi: 10.1109/ACCESS.2020.3026193](https://doi.org/10.1109/ACCESS.2020.3026193).

# Instructions

## ABB

Create an ABB IRB 1600 controller in ABB RobotStudio and load "Module1.mod" as a RAPID code module. Run the module.

## MATLAB

1. Import all the files in the same folder.
2. Run "a_setup.m".

**Note:** The ABB controller RAPID code must be executed before running "a_setup.m".

### Use of the A* control:

- Control the robot with A* using the command "fm":
```
>> fm
```
Input each coordinate when requested.
- Control the robot with diagonal-A* using the command "fmd":
```
>> fmd
```
Input each coordinate when requested. This allows the robot TCP to move diagonally and skip adjacent cells.

### Manual commands:
Go up one cell:
```
>> gu
```
Go down one cell:
```
>> gd
```
Go one cell towards the human facing the robot:
```
>> gc
```
Go one cell backwards from the human facing the robot:
```
>> gb
```
Go left one cell:

```
>> gl
```
Go right one cell:
```
>> gr
```
Change orientation UP:
```
>> ou
```
Change orientation DOWN:
```
>> od
```
Change orientation CENTER:
```
>> oc
```
Change orientation LEFT:
```
>> ol
```
Change orientation RIGHT:
```
>> oor
```
Go to home position with A*:
```
>> gh
```
Go to home position with diagonal-A* (moving diagonally too):
```
>> ghd
```

Note that commands can be composed using a semicolon, for example:
```
>> gu;gu;gl;gl;gb;gb;gd;gd;gr;gr;gc;gc;
```
