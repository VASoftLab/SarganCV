#ifndef UDPPACKET_H
#define UDPPACKET_H

#include <cstdint>
#include <QByteArray>
#include <QDataStream>

class UDPPacket
{
public:
    UDPPacket();
    std::int8_t udpCMD;
    float udpANG;
    QByteArray toByteArray();
};

#endif // UDPPACKET_H
