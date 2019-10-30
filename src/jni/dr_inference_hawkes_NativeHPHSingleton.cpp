#include <memory>
#include <vector>
#include <iostream>

// #include "Hawkes.hpp"
#include "NewHawkes.hpp"
#include "dr_inference_hawkes_NativeHPHSingleton.h"

typedef std::shared_ptr<hph::AbstractHawkes> InstancePtr;
std::vector<InstancePtr> instances;

extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_hawkes_NativeHPHSingleton_initialize
  (JNIEnv *, jobject, jint embeddingDimension, jint elementCount, jlong flags, jint device, jint threads) {
    instances.emplace_back(
//         std::make_shared<hph::Hawkes<double>>(embeddingDimension, elementCount, flags));
//         std::make_shared<hph::NewHawkes<double,hph::CpuAccumulate>>(embeddingDimension, elementCount, flags)
		hph::factory(embeddingDimension, elementCount, flags, device, threads)
    );
    return instances.size() - 1;
  }

//extern "C"
//JNIEXPORT void JNICALL Java_dr_inference_hawkes_NativeHPHSingleton_updateLocations
//  (JNIEnv *env, jobject, jint instance, jint index, jdoubleArray xArray) {
//  	jsize len = env->GetArrayLength(xArray);
//  	jdouble* x = env->GetDoubleArrayElements(xArray, NULL);
//
//    instances[instance]->updateLocations(index, x, len);
//
//    env->ReleaseDoubleArrayElements(xArray, x, JNI_ABORT);
//}

extern "C"
JNIEXPORT jdouble JNICALL Java_dr_inference_hawkes_NativeHPHSingleton_getSumOfIncrements
  (JNIEnv *, jobject, jint instance) {
    return instances[instance]->getSumOfLikContribs();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_hawkes_NativeHPHSingleton_storeState
  (JNIEnv *, jobject, jint instance) {
    instances[instance]->storeState();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_hawkes_NativeHPHSingleton_restoreState
  (JNIEnv *, jobject, jint instance) {
    instances[instance]->restoreState();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_hawkes_NativeHPHSingleton_acceptState
  (JNIEnv *, jobject, jint instance) {
    instances[instance]->acceptState();
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_hawkes_NativeHPHSingleton_setLocDistsData
  (JNIEnv *env, jobject, jint instance, jdoubleArray xArray) {
  	jsize len = env->GetArrayLength(xArray);
  	jdouble* x = env->GetDoubleArrayElements(xArray, NULL);

    instances[instance]->setLocDistsData(x, len);

    env->ReleaseDoubleArrayElements(xArray, x, JNI_ABORT);
}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_hawkes_NativeHPHSingleton_setTimDiffsData
        (JNIEnv *env, jobject, jint instance, jdoubleArray xArray) {
    jsize len = env->GetArrayLength(xArray);
    jdouble* x = env->GetDoubleArrayElements(xArray, NULL);

    instances[instance]->setTimDiffsData(x, len);

    env->ReleaseDoubleArrayElements(xArray, x, JNI_ABORT);
}

//extern "C"
//JNIEXPORT void JNICALL Java_dr_inference_hawkes_NativeHPHSingleton_getLocationGradient
//  (JNIEnv *env, jobject, jint instance, jdoubleArray xArray) {
//	jsize len = env->GetArrayLength(xArray);
//	jdouble* x = env->GetDoubleArrayElements(xArray, NULL); // TODO: Try GetPrimitiveArrayCritical
//
//	instances[instance]->getLogLikelihoodGradient(x, len);
//
//	env->ReleaseDoubleArrayElements(xArray, x, 0); // copy values back
//}

extern "C"
JNIEXPORT void JNICALL Java_dr_inference_hawkes_NativeHPHSingleton_setParameters
  (JNIEnv *env, jobject, jint instance, jdoubleArray xArray) {
  	jsize len = env->GetArrayLength(xArray);
  	jdouble* x = env->GetDoubleArrayElements(xArray, NULL);

    instances[instance]->setParameters(x, len);

    env->ReleaseDoubleArrayElements(xArray, x, JNI_ABORT);
}

extern "C"
JNIEXPORT jint JNICALL Java_dr_inference_hawkes_NativeHPHSingleton_getInternalDimension
        (JNIEnv *, jobject, jint instance) {
    return instances[instance]->getInternalDimension();
}


// jsize len = (*env)->GetArrayLength(env, arr);
//     jdouble *partials = env->GetDoubleArrayElements(inPartials, NULL);
//
// 	jint errCode = (jint)beagleSetPartials(instance, bufferIndex, (double *)partials);
//
//     env->ReleaseDoubleArrayElements(inPartials, partials, JNI_ABORT);
//         // not using JNI_ABORT flag here because we want the values to be copied back...
//     env->ReleaseDoubleArrayElements(outPartials, partials, 0);
