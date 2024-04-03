#include <iostream>
#include <sstream>
#include <fstream>


using namespace std;


#ifndef LEGGIFILE_H
#define LEGGIFILE_H

/* funzioni per leggere argv */
template <class T> bool SRead (const  string & s, T & val){
        istringstream ss(s);
        ss>>val;
        return (bool)ss;
};
template <class T> bool SPrint (string & s, const T & val){
        ostringstream ss(s);
        ss<<val;
        s=ss.str();
        return (bool)ss;
};
void TestSRP(){
        int test; SRead("125",test);cout<<test<<endl;
        string s; SPrint(s,test);cout<<s<<endl;

}
/* fine */

/* width */
ostream& W10(ostream & stream){
	stream.width(10);
	stream.flags( ios::left );
	return stream;
}
template <int I>ostream&   W(ostream & stream){
	stream.width(I);
	stream.flags( ios::left );
	return stream;
}
template <class T> struct WC{
	int w;
	T local;
	std::_Ios_Fmtflags fl;
	WC(int w_,T local_,std::_Ios_Fmtflags fl_):w(w_),local(local_),fl(fl_){}
	ostream & operator()(ostream & stream)const{
		stream.width(w);
		stream.flags( fl );
		stream<<local;
		return stream;
	}
};
template <class T> ostream & operator<< (ostream & stream,const WC<T> & fun){
	return fun(stream);
}

template <class T> WC<T> WL(int w_,T local_){
	return WC<T>( w_,local_,ios::left);
}
template <class T> WC<T> WR(int w_,T local_){
	return WC<T >( w_,local_,ios::right);
}

#endif
