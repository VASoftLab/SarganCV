#include "udppacket.h"

UDPPacket::UDPPacket()
{
    udpCMD = -1;
    udpANG = 0;
}

QByteArray UDPPacket::toByteArray()
{
    QByteArray result;
    QDataStream stream(&result, QIODeviceBase::WriteOnly);
    stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
    // stream.setByteOrder(QDataStream::LittleEndian);
    stream.setVersion(QDataStream::Qt_6_3);

    stream << udpCMD;
    stream << udpANG;

    return result;
}
